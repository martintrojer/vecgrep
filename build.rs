use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const MODEL_FILE: &str = "model.onnx";
const TOKENIZER_FILE: &str = "tokenizer.json";

const MODEL_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx";
const TOKENIZER_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json";

// SHA-256 hashes for verification (empty = skip verification for initial bootstrap)
const MODEL_SHA256: &str = "";
const TOKENIZER_SHA256: &str = "";

fn cache_dir() -> PathBuf {
    if let Ok(dir) = env::var("VECGREP_MODEL_CACHE") {
        return PathBuf::from(dir);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("vecgrep")
        .join("models")
}

fn download_file(
    url: &str,
    dest: &Path,
    expected_sha256: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if dest.exists() {
        if expected_sha256.is_empty() {
            println!("cargo:warning=Using cached {}", dest.display());
            return Ok(());
        }
        // Verify existing file
        let data = fs::read(dest)?;
        let hash = hex::encode(Sha256::digest(&data));
        if hash == expected_sha256 {
            println!(
                "cargo:warning=Using cached {} (hash verified)",
                dest.display()
            );
            return Ok(());
        }
        println!(
            "cargo:warning=Hash mismatch for {}, re-downloading",
            dest.display()
        );
    }

    println!("cargo:warning=Downloading {} ...", url);
    let response = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?
        .get(url)
        .send()?;

    if !response.status().is_success() {
        return Err(format!("HTTP {} for {}", response.status(), url).into());
    }

    let bytes = response.bytes()?;

    if !expected_sha256.is_empty() {
        let hash = hex::encode(Sha256::digest(&bytes));
        if hash != expected_sha256 {
            return Err(format!(
                "SHA-256 mismatch for {}: expected {}, got {}",
                url, expected_sha256, hash
            )
            .into());
        }
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(dest)?;
    file.write_all(&bytes)?;

    println!(
        "cargo:warning=Downloaded {} ({} bytes)",
        dest.display(),
        bytes.len()
    );
    Ok(())
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let model_dir = out_dir.join("models");
    fs::create_dir_all(&model_dir).expect("Failed to create models dir");

    let cache = cache_dir();
    fs::create_dir_all(&cache).expect("Failed to create cache dir");

    // Download model
    let cached_model = cache.join(MODEL_FILE);
    download_file(MODEL_URL, &cached_model, MODEL_SHA256).expect("Failed to download model");

    // Download tokenizer
    let cached_tokenizer = cache.join(TOKENIZER_FILE);
    download_file(TOKENIZER_URL, &cached_tokenizer, TOKENIZER_SHA256)
        .expect("Failed to download tokenizer");

    // Copy to OUT_DIR for include_bytes!
    let dest_model = model_dir.join(MODEL_FILE);
    let dest_tokenizer = model_dir.join(TOKENIZER_FILE);

    fs::copy(&cached_model, &dest_model).expect("Failed to copy model to OUT_DIR");
    fs::copy(&cached_tokenizer, &dest_tokenizer).expect("Failed to copy tokenizer to OUT_DIR");

    // Rerun only if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VECGREP_MODEL_CACHE");
}
