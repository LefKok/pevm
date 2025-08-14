//! Benchmark fixed-size blocks (10,000 transactions).

// TODO: More fancy benchmarks & plots.

use std::{num::NonZeroUsize, sync::Arc, thread, time::Instant};

use alloy_primitives::{keccak256, Address, U160, U256};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier};
use pevm::{
    chain::PevmEthereum,
    execute_revm_sequential,
    Bytecodes,
    ChainState,
    EvmAccount,
    InMemoryStorage,
    Pevm,
};
use rayon::prelude::*;
use revm::primitives::{BlockEnv, SpecId, TransactTo, TxEnv};

// Better project structure

/// common module
#[path = "../tests/common/mod.rs"]
pub mod common;

/// erc20 module
#[path = "../tests/erc20/mod.rs"]
pub mod erc20;

/// uniswap module
#[path = "../tests/uniswap/mod.rs"]
pub mod uniswap;

/// Fixed transaction count per experiment
const BLOCK_SIZE: usize = 1;

/// Signed transaction wrapper (signature to be filled in next steps)
struct SignedTx {
    tx: TxEnv,
    // Ed25519 signature bytes (64 bytes)
    signature: Vec<u8>,
}

struct SimpleSigner {
    sk: SigningKey,
}

impl SimpleSigner {
    fn new_deterministic() -> Self {
        // Fixed private key for reproducibility in the benchmark
        let sk_bytes = [7u8; 32];
        let sk = SigningKey::from_bytes(&sk_bytes);
        Self { sk }
    }

    fn sign_hash(&self, hash32: [u8; 32]) -> [u8; 64] {
        let sig: Signature = self.sk.sign(&hash32);
        sig.to_bytes()
    }
}

// Serialization helpers for TxEnv to simulate client<->sequencer overhead.
fn u256_to_be_bytes(value: U256) -> [u8; 32] {
    value.to_be_bytes::<32>()
}

fn u256_from_be_slice(slice: &[u8]) -> U256 {
    debug_assert_eq!(slice.len(), 32);
    U256::from_be_slice(slice)
}

fn serialize_txenv(tx: &TxEnv) -> Vec<u8> {
    let mut out = Vec::with_capacity(128 + tx.data.len());

    // version
    out.push(1u8);

    // caller (20 bytes)
    out.extend_from_slice(tx.caller.as_slice());

    // transact_to tag + address (we only support Call in this bench)
    match tx.transact_to {
        TransactTo::Call(to) => {
            out.push(1u8);
            out.extend_from_slice(to.as_slice());
        }
        TransactTo::Create => {
            out.push(0u8);
            // placeholder for address bytes to keep layout simple
            out.extend_from_slice(&[0u8; 20]);
        }
    }

    // gas_limit (u64)
    out.extend_from_slice(&tx.gas_limit.to_be_bytes());

    // gas_price (U256)
    out.extend_from_slice(&u256_to_be_bytes(tx.gas_price));

    // value (U256)
    out.extend_from_slice(&u256_to_be_bytes(tx.value));

    // nonce (Option<u64>)
    if let Some(nonce) = tx.nonce {
        out.push(1u8);
        out.extend_from_slice(&nonce.to_be_bytes());
    } else {
        out.push(0u8);
    }

    // data length (u32) + data bytes
    let data: &[u8] = tx.data.as_ref();
    let len: u32 = data.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(data);

    out
}

fn serialize_signed_tx(signed: &SignedTx) -> Vec<u8> {
    let mut out = serialize_txenv(&signed.tx);
    // append signature length (u8) and bytes
    let sig_len = u8::try_from(signed.signature.len()).unwrap_or(0);
    out.push(sig_len);
    out.extend_from_slice(&signed.signature);
    out
}

fn deserialize_txenv(mut bytes: &[u8]) -> Result<TxEnv, &'static str> {
    use std::convert::TryInto;

    // version
    if bytes.len() < 1 {
        return Err("buffer too small");
    }
    let _version = bytes[0];
    bytes = &bytes[1..];

    // caller
    if bytes.len() < 20 {
        return Err("missing caller");
    }
    let caller = Address::from_slice(&bytes[..20]);
    bytes = &bytes[20..];

    // transact_to tag + address
    if bytes.len() < 1 + 20 {
        return Err("missing transact_to");
    }
    let tag = bytes[0];
    let to_addr = Address::from_slice(&bytes[1..21]);
    bytes = &bytes[21..];
    let transact_to = match tag {
        1 => TransactTo::Call(to_addr),
        0 => return Err("TransactTo::Create not supported in serde bench"),
        _ => return Err("invalid transact_to tag"),
    };

    // gas_limit
    if bytes.len() < 8 {
        return Err("missing gas_limit");
    }
    let gas_limit = u64::from_be_bytes(bytes[..8].try_into().unwrap());
    bytes = &bytes[8..];

    // gas_price
    if bytes.len() < 32 {
        return Err("missing gas_price");
    }
    let gas_price = u256_from_be_slice(&bytes[..32]);
    bytes = &bytes[32..];

    // value
    if bytes.len() < 32 {
        return Err("missing value");
    }
    let value = u256_from_be_slice(&bytes[..32]);
    bytes = &bytes[32..];

    // nonce
    if bytes.is_empty() {
        return Err("missing nonce tag");
    }
    let has_nonce = bytes[0] == 1u8;
    bytes = &bytes[1..];
    let nonce = if has_nonce {
        if bytes.len() < 8 {
            return Err("missing nonce value");
        }
        let n = u64::from_be_bytes(bytes[..8].try_into().unwrap());
        bytes = &bytes[8..];
        Some(n)
    } else {
        None
    };

    // data
    if bytes.len() < 4 {
        return Err("missing data length");
    }
    let len = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
    bytes = &bytes[4..];
    if bytes.len() < len {
        return Err("missing data bytes");
    }
    let data = bytes[..len].to_vec();

    Ok(TxEnv {
        caller,
        transact_to,
        value,
        gas_limit,
        gas_price,
        data: data.into(),
        nonce,
        ..TxEnv::default()
    })
}

fn deserialize_signed_tx(mut bytes: &[u8]) -> Result<SignedTx, &'static str> {
    // First parse TxEnv
    let tx = deserialize_txenv(bytes)?;

    // Re-encode to know how many bytes were consumed by tx to locate signature
    let encoded_tx = serialize_txenv(&tx);
    if bytes.len() < encoded_tx.len() + 1 {
        return Err("missing signature length");
    }
    let sig_len = bytes[encoded_tx.len()] as usize;
    bytes = &bytes[encoded_tx.len() + 1..];
    if bytes.len() < sig_len {
        return Err("missing signature bytes");
    }
    let signature = bytes[..sig_len].to_vec();
    Ok(SignedTx { tx, signature })
}

fn client_sign_txs(txs: &[TxEnv]) -> Vec<Vec<u8>> {
    let signer = SimpleSigner::new_deterministic();
    txs.iter()
        .map(|tx| {
            let tx_bytes = serialize_txenv(tx);
            let hash = keccak256(&tx_bytes);
            let sig = signer.sign_hash(hash.0);
            serialize_signed_tx(&SignedTx {
                tx: tx.clone(),
                signature: sig.to_vec(),
            })
        })
        .collect()
}

fn verify_signed_tx(signed: &SignedTx) -> Result<(), &'static str> {
    // Simulate verification cost by verifying with the deterministic verifying key
    let tx_bytes = serialize_txenv(&signed.tx);
    let hash = keccak256(&tx_bytes);
    let sk_bytes = [7u8; 32];
    let vk = SigningKey::from_bytes(&sk_bytes).verifying_key();
    let sig = Signature::from_slice(&signed.signature).map_err(|_| "bad sig")?;
    vk.verify(&hash.0, &sig).map_err(|_| "verify fail")
}

fn verify_signed_txs_to_txs(serialized: &[Vec<u8>]) -> Result<Vec<TxEnv>, &'static str> {
    let mut out = Vec::with_capacity(serialized.len());
    for b in serialized {
        let s = deserialize_signed_tx(b)?;
        verify_signed_tx(&s)?;
        out.push(s.tx);
    }
    Ok(out)
}

fn verify_signed_txs_to_txs_parallel(serialized: &[Vec<u8>]) -> Result<Vec<TxEnv>, &'static str> {
    serialized
        .par_iter()
        .map(|b| {
            let s = deserialize_signed_tx(b)?;
            verify_signed_tx(&s)?;
            Ok::<TxEnv, &'static str>(s.tx)
        })
        .collect()
}

#[cfg(feature = "global-alloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Runs a benchmark for executing a set of transactions on a given blockchain state.
pub fn bench(c: &mut Criterion, name: &str, storage: InMemoryStorage, txs: Vec<TxEnv>) {
    let concurrency_level = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let chain = PevmEthereum::mainnet();
    let spec_id = SpecId::LATEST;
    let block_env = BlockEnv::default();
    let mut pevm = Pevm::default();
    let mut group = c.benchmark_group(name);
    // Report TPS via Criterion throughput (transactions per iteration)
    group.throughput(Throughput::Elements(txs.len() as u64));

    // Precompute serialized bytes (simulate client-side serialization)
    let pre_serialized = Arc::new(client_sign_txs(&txs));

    // One-off TPS print including deserialization + execution (serialization excluded)
    {
        // Sequential: measure deserialize + execute
        let start = Instant::now();
        let decoded = verify_signed_txs_to_txs(&pre_serialized).expect("verify signed txs");
        let _ = execute_revm_sequential(&chain, &storage, spec_id, block_env.clone(), decoded);
        let elapsed = start.elapsed();
        let tps = (txs.len() as f64) / elapsed.as_secs_f64();
        println!(
            "{}: Sequential TPS ({} tx, deser+verify): {:.2}",
            name,
            txs.len(),
            tps
        );

        // Parallel: measure deserialize (parallel verify) + execute
        let start_p = Instant::now();
        let decoded_p =
            verify_signed_txs_to_txs_parallel(&pre_serialized).expect("verify signed txs");
        let _ = pevm.execute_revm_parallel(
            &chain,
            &storage,
            spec_id,
            block_env.clone(),
            decoded_p,
            concurrency_level,
        );
        let elapsed_p = start_p.elapsed();
        let tps_p = (txs.len() as f64) / elapsed_p.as_secs_f64();
        println!(
            "{}: Parallel TPS ({} tx, deser+verify): {:.2}",
            name,
            txs.len(),
            tps_p
        );
    }
    group.bench_function("Sequential", |b| {
        let pre_serialized = pre_serialized.clone();
        b.iter(|| {
            let decoded = verify_signed_txs_to_txs(black_box(pre_serialized.as_ref()))
                .expect("verify signed txs");
            execute_revm_sequential(
                black_box(&chain),
                black_box(&storage),
                black_box(spec_id),
                black_box(block_env.clone()),
                black_box(decoded),
            )
        })
    });
    group.bench_function("Parallel", |b| {
        let pre_serialized = pre_serialized.clone();
        b.iter(|| {
            let decoded = verify_signed_txs_to_txs_parallel(black_box(pre_serialized.as_ref()))
                .expect("verify signed txs");
            pevm.execute_revm_parallel(
                black_box(&chain),
                black_box(&storage),
                black_box(spec_id),
                black_box(block_env.clone()),
                black_box(decoded),
                black_box(concurrency_level),
            )
        })
    });
    group.finish();
}

/// Benchmarks the execution time of raw token transfers.
pub fn bench_raw_transfers(c: &mut Criterion) {
    // Skip the built-in precompiled contracts addresses.
    const START_ADDRESS: usize = 1000;
    const MINER_ADDRESS: usize = 0;
    let storage = InMemoryStorage::new(
        std::iter::once(MINER_ADDRESS)
            .chain(START_ADDRESS..START_ADDRESS + BLOCK_SIZE)
            .map(common::mock_account)
            .collect(),
        Default::default(),
        Default::default(),
    );
    bench(
        c,
        "Independent Raw Transfers",
        storage,
        (0..BLOCK_SIZE)
            .map(|i| {
                let address = Address::from(U160::from(START_ADDRESS + i));
                TxEnv {
                    caller: address,
                    transact_to: TransactTo::Call(address),
                    value: U256::from(1),
                    gas_limit: common::RAW_TRANSFER_GAS_LIMIT,
                    gas_price: U256::from(1),
                    ..TxEnv::default()
                }
            })
            .collect::<Vec<_>>(),
    );
}

/// Benchmarks the execution time of ERC-20 token transfers.
pub fn bench_erc20(c: &mut Criterion) {
    let (mut state, bytecodes, txs) = erc20::generate_cluster(BLOCK_SIZE, 1, 1);
    state.insert(Address::ZERO, EvmAccount::default()); // Beneficiary
    bench(
        c,
        "Independent ERC20",
        InMemoryStorage::new(state, Arc::new(bytecodes), Default::default()),
        txs,
    );
}

/// Benchmarks the execution time of Uniswap V3 swap transactions.
pub fn bench_uniswap(c: &mut Criterion) {
    let mut final_state = ChainState::from_iter([(Address::ZERO, EvmAccount::default())]); // Beneficiary
    let mut final_bytecodes = Bytecodes::default();
    let mut final_txs = Vec::<TxEnv>::new();
    for _ in 0..BLOCK_SIZE {
        let (state, bytecodes, txs) = uniswap::generate_cluster(1, 1);
        final_state.extend(state);
        final_bytecodes.extend(bytecodes);
        final_txs.extend(txs);
    }
    bench(
        c,
        "Independent Uniswap",
        InMemoryStorage::new(final_state, Arc::new(final_bytecodes), Default::default()),
        final_txs,
    );
}

/// Runs a series of benchmarks to evaluate the performance of different transaction types.
pub fn benchmark_gigagas(c: &mut Criterion) {
    bench_raw_transfers(c);
    bench_erc20(c);
    bench_uniswap(c);
}

// HACK: we can't document public items inside of the macro
#[allow(missing_docs)]
mod benches {
    use super::*;
    criterion_group!(benches, benchmark_gigagas);
}

criterion_main!(benches::benches);
