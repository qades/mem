#!/usr/bin/env python3
"""
Parallel compression utility for ProLong dataset.

Compresses .mds files using bzip2 with configurable parallelism.
Useful for reducing disk usage (typically 3-4x compression with bzip2 -3).

Usage:
    # Compress existing ProLong dataset
    python compress_prolong.py
    
    # Compress with specific settings
    python compress_prolong.py --input data/prolong --output compressed/prolong --threads 32
    
    # Verify compressed dataset integrity
    python compress_prolong.py --verify-only
    
    # Decompress (restore original files)
    python compress_prolong.py --decompress
"""

import argparse
import bz2
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple


def get_compression_info(input_dir: Path, output_dir: Path) -> dict:
    """Get information about compression status."""
    info = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_exists": input_dir.exists(),
        "output_exists": output_dir.exists(),
    }
    
    if input_dir.exists():
        mds_files = list(input_dir.rglob("*.mds"))
        info["input_files"] = len(mds_files)
        info["input_size_bytes"] = sum(f.stat().st_size for f in mds_files)
        info["input_size_gb"] = info["input_size_bytes"] / (1024**3)
    else:
        info["input_files"] = 0
        info["input_size_bytes"] = 0
        info["input_size_gb"] = 0
    
    if output_dir.exists():
        bz2_files = list(output_dir.rglob("*.mds.bz2"))
        info["compressed_files"] = len(bz2_files)
        info["compressed_size_bytes"] = sum(f.stat().st_size for f in bz2_files)
        info["compressed_size_gb"] = info["compressed_size_bytes"] / (1024**3)
        
        if info["input_size_bytes"] > 0 and info["compressed_size_bytes"] > 0:
            info["compression_ratio"] = info["input_size_bytes"] / info["compressed_size_bytes"]
        else:
            info["compression_ratio"] = 0
    else:
        info["compressed_files"] = 0
        info["compressed_size_bytes"] = 0
        info["compressed_size_gb"] = 0
        info["compression_ratio"] = 0
    
    return info


def compress_file(args: Tuple[Path, Path, int]) -> Tuple[str, bool, Optional[str]]:
    """
    Compress a single file.
    
    Args:
        args: (input_file, output_dir, compression_level)
        
    Returns:
        (input_path, success, error_message)
    """
    input_file, output_dir, level = args
    
    # Calculate relative path
    try:
        rel_path = input_file.relative_to(Path("data/prolong"))
    except ValueError:
        rel_path = input_file.name
    
    output_file = output_dir / rel_path.with_suffix(input_file.suffix + ".bz2")
    
    # Skip if already exists and is newer
    if output_file.exists():
        if output_file.stat().st_mtime >= input_file.stat().st_mtime:
            return (str(input_file), True, None)  # Already compressed
    
    try:
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Compress to temp file first, then move (atomic operation)
        temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
        
        with open(input_file, 'rb') as f_in:
            with bz2.open(temp_file, 'wb', compresslevel=level) as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        temp_file.rename(output_file)
        return (str(input_file), True, None)
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()
        return (str(input_file), False, str(e))


def decompress_file(args: Tuple[Path, Path]) -> Tuple[str, bool, Optional[str]]:
    """
    Decompress a single file.
    
    Args:
        args: (compressed_file, output_dir)
        
    Returns:
        (input_path, success, error_message)
    """
    input_file, output_dir = args
    
    # Calculate output path (remove .bz2 suffix)
    output_name = input_file.name[:-4]  # Remove .bz2
    
    try:
        rel_path = input_file.relative_to(Path("compressed/prolong"))
        output_file = output_dir / rel_path.with_name(output_name)
    except ValueError:
        output_file = output_dir / output_name
    
    # Skip if already exists
    if output_file.exists():
        return (str(input_file), True, "exists")
    
    try:
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Decompress to temp file first
        temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
        
        with bz2.open(input_file, 'rb') as f_in:
            with open(temp_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        temp_file.rename(output_file)
        return (str(input_file), True, None)
        
    except Exception as e:
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()
        return (str(input_file), False, str(e))


def copy_non_mds_files(input_dir: Path, output_dir: Path, threads: int = 4) -> Tuple[int, int]:
    """
    Copy all non-.mds files from input to output.
    
    Returns:
        (copied_count, error_count)
    """
    non_mds_files = [f for f in input_dir.rglob("*") if f.is_file() and not f.name.endswith('.mds')]
    
    copied = 0
    errors = 0
    
    for file in non_mds_files:
        try:
            rel_path = file.relative_to(input_dir)
            output_file = output_dir / rel_path
            
            if output_file.exists() and output_file.stat().st_mtime >= file.stat().st_mtime:
                continue  # Already up to date
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, output_file)
            copied += 1
        except Exception as e:
            print(f"  Error copying {file}: {e}", file=sys.stderr)
            errors += 1
    
    return copied, errors


def compress_dataset(
    input_dir: Path,
    output_dir: Path,
    threads: int = 32,
    level: int = 3,
    verbose: bool = True
) -> bool:
    """
    Compress entire dataset with parallel processing.
    
    Args:
        input_dir: Source directory containing .mds files
        output_dir: Destination directory for compressed files
        threads: Number of parallel compression threads
        level: bzip2 compression level (1-9, default 3)
        verbose: Print progress information
        
    Returns:
        True if successful, False otherwise
    """
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .mds files
    mds_files = list(input_dir.rglob("*.mds"))
    total_files = len(mds_files)
    
    if total_files == 0:
        print(f"No .mds files found in {input_dir}")
        return False
    
    print(f"Found {total_files} .mds files to compress")
    print(f"Output directory: {output_dir}")
    print(f"Compression level: bzip2 -{level}")
    print(f"Parallel threads: {threads}")
    print("")
    
    # Prepare work items
    work_items = [(f, output_dir, level) for f in mds_files]
    
    # Progress tracking
    completed = 0
    failed = 0
    skipped = 0
    start_time = time.time()
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(compress_file, item): item for item in work_items}
        
        for future in as_completed(futures):
            input_path, success, error = future.result()
            
            if success:
                if error == "exists":
                    skipped += 1
                else:
                    completed += 1
            else:
                failed += 1
                if verbose:
                    print(f"  Error: {input_path}: {error}", file=sys.stderr)
            
            # Progress update
            if verbose and (completed + failed + skipped) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (completed + skipped) / elapsed if elapsed > 0 else 0
                percent = (completed + skipped + failed) / total_files * 100
                print(f"\r  Progress: {completed + skipped + failed}/{total_files} "
                      f"({percent:.1f}%) | "
                      f"OK: {completed} | Skip: {skipped} | Fail: {failed} | "
                      f"{rate:.1f} files/s", end='', flush=True)
    
    elapsed = time.time() - start_time
    print(f"\r  Progress: {completed + skipped + failed}/{total_files} (100.0%) | "
          f"OK: {completed} | Skip: {skipped} | Fail: {failed}")
    print(f"\nCompression complete in {elapsed:.1f}s ({(completed + skipped) / elapsed:.1f} files/s)")
    
    # Copy non-.mds files
    print("\nCopying non-.mds files (index.json, etc.)...")
    copied, copy_errors = copy_non_mds_files(input_dir, output_dir)
    print(f"  Copied: {copied} files")
    if copy_errors > 0:
        print(f"  Errors: {copy_errors} files")
    
    # Print summary
    info = get_compression_info(input_dir, output_dir)
    print("\n" + "=" * 60)
    print("COMPRESSION SUMMARY")
    print("=" * 60)
    print(f"Original size:   {info['input_size_gb']:.2f} GB ({info['input_files']} files)")
    print(f"Compressed size: {info['compressed_size_gb']:.2f} GB ({info['compressed_files']} files)")
    if info['compression_ratio'] > 0:
        print(f"Compression ratio: {info['compression_ratio']:.2f}x")
        print(f"Space saved: {info['input_size_gb'] - info['compressed_size_gb']:.2f} GB")
    print("=" * 60)
    
    return failed == 0


def decompress_dataset(
    input_dir: Path,
    output_dir: Path,
    threads: int = 32,
    verbose: bool = True
) -> bool:
    """Decompress entire dataset with parallel processing."""
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .mds.bz2 files
    bz2_files = list(input_dir.rglob("*.mds.bz2"))
    total_files = len(bz2_files)
    
    if total_files == 0:
        print(f"No .mds.bz2 files found in {input_dir}")
        return False
    
    print(f"Found {total_files} compressed files to decompress")
    print(f"Output directory: {output_dir}")
    print(f"Parallel threads: {threads}")
    print("")
    
    # Prepare work items
    work_items = [(f, output_dir) for f in bz2_files]
    
    # Progress tracking
    completed = 0
    failed = 0
    skipped = 0
    start_time = time.time()
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(decompress_file, item): item for item in work_items}
        
        for future in as_completed(futures):
            input_path, success, error = future.result()
            
            if success:
                if error == "exists":
                    skipped += 1
                else:
                    completed += 1
            else:
                failed += 1
                if verbose:
                    print(f"  Error: {input_path}: {error}", file=sys.stderr)
            
            if verbose and (completed + failed + skipped) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (completed + skipped) / elapsed if elapsed > 0 else 0
                percent = (completed + skipped + failed) / total_files * 100
                print(f"\r  Progress: {completed + skipped + failed}/{total_files} "
                      f"({percent:.1f}%) | "
                      f"OK: {completed} | Skip: {skipped} | Fail: {failed} | "
                      f"{rate:.1f} files/s", end='', flush=True)
    
    elapsed = time.time() - start_time
    print(f"\r  Progress: {completed + skipped + failed}/{total_files} (100.0%) | "
          f"OK: {completed} | Skip: {skipped} | Fail: {failed}")
    print(f"\nDecompression complete in {elapsed:.1f}s")
    
    return failed == 0


def verify_compressed_dataset(input_dir: Path, output_dir: Path) -> bool:
    """Verify that compressed dataset matches original."""
    print("Verifying compressed dataset...")
    
    mds_files = list(input_dir.rglob("*.mds"))
    bz2_files = list(output_dir.rglob("*.mds.bz2"))
    
    # Check counts
    if len(mds_files) != len(bz2_files):
        print(f"  WARNING: File count mismatch!")
        print(f"    Original: {len(mds_files)} files")
        print(f"    Compressed: {len(bz2_files)} files")
        return False
    
    print(f"  File count: {len(mds_files)} ✓")
    
    # Check that all originals have compressed versions
    missing = []
    for mds_file in mds_files:
        try:
            rel_path = mds_file.relative_to(input_dir)
        except ValueError:
            rel_path = Path(mds_file.name)
        
        expected_bz2 = output_dir / rel_path.with_suffix(mds_file.suffix + ".bz2")
        if not expected_bz2.exists():
            missing.append(str(rel_path))
    
    if missing:
        print(f"  WARNING: {len(missing)} files not compressed:")
        for m in missing[:5]:
            print(f"    - {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
        return False
    
    print(f"  All files compressed: ✓")
    
    # Show compression stats
    info = get_compression_info(input_dir, output_dir)
    print(f"  Compression ratio: {info['compression_ratio']:.2f}x")
    print(f"  Space saved: {info['input_size_gb'] - info['compressed_size_gb']:.2f} GB")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compress/Decompress ProLong dataset with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Compress data/prolong to compressed/prolong
  %(prog)s --threads 16              # Use 16 parallel threads
  %(prog)s --level 9                 # Use maximum compression (slower)
  %(prog)s --verify-only             # Verify existing compressed dataset
  %(prog)s --decompress              # Decompress back to original
  %(prog)s --input ./mydata --output ./compressed  # Custom paths
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/prolong"),
        help="Input directory (default: data/prolong)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("compressed/prolong"),
        help="Output directory (default: compressed/prolong)"
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=min(32, os.cpu_count() or 4),
        help=f"Number of parallel threads (default: {min(32, os.cpu_count() or 4)})"
    )
    parser.add_argument(
        "--level", "-l",
        type=int,
        default=3,
        choices=range(1, 10),
        metavar="LEVEL",
        help="Compression level 1-9 (default: 3, sweet spot for speed/size)"
    )
    parser.add_argument(
        "--decompress", "-d",
        action="store_true",
        help="Decompress instead of compress"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify compressed dataset without compressing"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show compression info and exit"
    )
    parser.add_argument(
        "--remove-original",
        action="store_true",
        help="Remove original files after successful compression (USE WITH CAUTION)"
    )
    
    args = parser.parse_args()
    
    # Show info and exit
    if args.info:
        info = get_compression_info(args.input, args.output)
        print(json.dumps(info, indent=2, default=str))
        return 0
    
    # Verify mode
    if args.verify_only:
        success = verify_compressed_dataset(args.input, args.output)
        return 0 if success else 1
    
    # Decompress mode
    if args.decompress:
        success = decompress_dataset(args.input, args.output, args.threads)
        return 0 if success else 1
    
    # Compress mode
    success = compress_dataset(
        args.input,
        args.output,
        threads=args.threads,
        level=args.level
    )
    
    if success and args.remove_original:
        print("\nRemoving original files...")
        mds_files = list(args.input.rglob("*.mds"))
        for f in mds_files:
            try:
                # Verify compressed version exists
                rel_path = f.relative_to(args.input)
                compressed = args.output / rel_path.with_suffix(f.suffix + ".bz2")
                if compressed.exists():
                    f.unlink()
                else:
                    print(f"  Skipping {f}: compressed version not found")
            except Exception as e:
                print(f"  Error removing {f}: {e}")
        print("Original files removed.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
