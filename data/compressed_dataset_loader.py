"""
Compressed dataset loader for ProLong and other MDS-format datasets.

Provides seamless access to .mds files that may be compressed (.mds.bz2) or
uncompressed (.mds). Falls back to compressed version if uncompressed is not available.
"""

import bz2
import json
import os
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union


class CompressedFileHandle:
    """
    File-like object that handles both compressed and uncompressed files.
    Automatically detects and decompresses .bz2 files on-the-fly.
    """
    
    def __init__(self, path: Union[str, Path], mode: str = 'rb'):
        """
        Open a file, automatically handling compression.
        
        Args:
            path: Path to file (may end in .bz2 or be uncompressed)
            mode: File mode (default: 'rb')
        """
        self.path = Path(path)
        self.mode = mode
        self._file: Optional[BinaryIO] = None
        self._is_compressed = False
        self._open()
    
    def _open(self):
        """Open the file, detecting compression automatically."""
        # Check if file exists as specified
        if self.path.exists():
            if str(self.path).endswith('.bz2'):
                self._file = bz2.open(self.path, self.mode)
                self._is_compressed = True
            else:
                self._file = open(self.path, self.mode)
        # Check for compressed version
        elif str(self.path).endswith('.mds') and Path(str(self.path) + '.bz2').exists():
            self._file = bz2.open(str(self.path) + '.bz2', self.mode)
            self._is_compressed = True
        # Check for uncompressed version
        elif str(self.path).endswith('.mds.bz2') and Path(str(self.path)[:-4]).exists():
            self.path = Path(str(self.path)[:-4])
            self._file = open(self.path, self.mode)
        else:
            raise FileNotFoundError(f"File not found: {self.path}")
    
    def read(self, size: int = -1) -> bytes:
        """Read bytes from file."""
        return self._file.read(size)
    
    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position in file."""
        return self._file.seek(offset, whence)
    
    def tell(self) -> int:
        """Get current file position."""
        return self._file.tell()
    
    def close(self):
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @property
    def is_compressed(self) -> bool:
        """Return True if reading from compressed file."""
        return self._is_compressed


def open_compressed(path: Union[str, Path], mode: str = 'rb') -> CompressedFileHandle:
    """
    Open a file, automatically handling bz2 compression.
    
    No flags needed - compression is detected automatically based on file
    extension. If you request 'file.mds' but only 'file.mds.bz2' exists,
    the compressed version is opened transparently.
    
    Args:
        path: Path to file (with or without .bz2 extension)
        mode: File mode (default: 'rb')
        
    Returns:
        CompressedFileHandle that behaves like a regular file object
        
    Example:
        >>> with open_compressed('data.mds', 'rb') as f:
        ...     data = f.read()
        >>> # Works seamlessly with both data.mds and data.mds.bz2!
    """
    return CompressedFileHandle(path, mode)


def find_shard_file(base_path: Union[str, Path], basename: str) -> Path:
    """
    Find a shard file, automatically checking for compressed version.
    
    No flags needed - automatically detects .bz2 extension. Prefers
    uncompressed if both exist.
    
    Args:
        base_path: Base directory path
        basename: File basename (e.g., 'shard.00000.mds')
        
    Returns:
        Path to the found file (uncompressed preferred if both exist)
        
    Raises:
        FileNotFoundError: If neither version exists
        
    Example:
        >>> # If only shard.00000.mds.bz2 exists, returns that path
        >>> path = find_shard_file('data/', 'shard.00000.mds')
        >>> # path is PosixPath('data/shard.00000.mds.bz2')
    """
    base_path = Path(base_path)
    uncompressed = base_path / basename
    compressed = base_path / (basename + '.bz2')
    
    # Prefer uncompressed if available
    if uncompressed.exists():
        return uncompressed
    elif compressed.exists():
        return compressed
    else:
        raise FileNotFoundError(f"Shard file not found: {uncompressed} (or .bz2)")


def load_mds_index(directory: Union[str, Path]) -> Dict:
    """
    Load MDS index.json file, handling both compressed and uncompressed datasets.
    
    The index.json is never compressed (it's small), but this function
    provides a consistent interface.
    
    Args:
        directory: Directory containing index.json
        
    Returns:
        Parsed index.json contents
    """
    directory = Path(directory)
    index_path = directory / 'index.json'
    
    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found in {directory}")
    
    with open(index_path, 'r') as f:
        return json.load(f)


def get_dataset_path(data_root: Union[str, Path] = "data/prolong") -> Path:
    """
    Get the path to the dataset, preferring compressed version if original doesn't exist.
    
    This is useful for seamlessly switching between compressed and uncompressed datasets.
    
    Args:
        data_root: Default path to dataset (default: "data/prolong")
        
    Returns:
        Path object pointing to the available dataset
    """
    data_root = Path(data_root)
    compressed_root = Path("compressed/prolong")
    
    # If original exists, use it
    if data_root.exists():
        return data_root
    # Otherwise use compressed
    elif compressed_root.exists():
        return compressed_root
    else:
        raise FileNotFoundError(f"Dataset not found at {data_root} or {compressed_root}")


def list_shard_files(directory: Union[str, Path]) -> List[Tuple[str, bool]]:
    """
    List all shard files in a directory, marking whether they're compressed.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of (filename, is_compressed) tuples
    """
    directory = Path(directory)
    shards = []
    
    # Find all .mds files
    for f in directory.glob('*.mds'):
        shards.append((f.name, False))
    
    # Find all .mds.bz2 files where .mds doesn't exist
    for f in directory.glob('*.mds.bz2'):
        uncompressed_name = f.name[:-4]  # Remove .bz2
        if not (directory / uncompressed_name).exists():
            shards.append((uncompressed_name, True))
    
    return sorted(shards)


def decompress_file(compressed_path: Union[str, Path], 
                   output_path: Optional[Union[str, Path]] = None,
                   remove_original: bool = False) -> Path:
    """
    Decompress a .bz2 file.
    
    Args:
        compressed_path: Path to .bz2 file
        output_path: Where to write decompressed file (default: remove .bz2 suffix)
        remove_original: Whether to delete the compressed file after decompression
        
    Returns:
        Path to decompressed file
    """
    compressed_path = Path(compressed_path)
    
    if output_path is None:
        if str(compressed_path).endswith('.bz2'):
            output_path = Path(str(compressed_path)[:-4])
        else:
            raise ValueError("output_path required if file doesn't end in .bz2")
    else:
        output_path = Path(output_path)
    
    # Decompress
    with bz2.open(compressed_path, 'rb') as src, open(output_path, 'wb') as dst:
        dst.write(src.read())
    
    if remove_original:
        compressed_path.unlink()
    
    return output_path


class StreamingDatasetWrapper:
    """
    Wrapper for mosaicml-streaming's StreamingDataset that handles compressed shards.
    
    This is a compatibility layer - if mosaicml-streaming is not installed,
    it provides basic file access to the shards.
    """
    
    def __init__(self, local: Union[str, Path], **kwargs):
        """
        Initialize dataset wrapper.
        
        Args:
            local: Local directory containing the dataset
            **kwargs: Additional arguments passed to StreamingDataset if available
        """
        self.local = Path(local)
        self.index = load_mds_index(self.local)
        self.shards = self.index.get('shards', [])
        
        # Try to import mosaicml-streaming for full functionality
        try:
            from streaming import StreamingDataset
            # Check if we need to handle compression
            has_compressed = any(
                (self.local / (s['raw_data']['basename'] + '.bz2')).exists()
                for s in self.shards
            )
            
            if has_compressed:
                # Need to decompress first for StreamingDataset
                self._temp_dir = self._decompress_shards()
                self.dataset = StreamingDataset(local=self._temp_dir, **kwargs)
            else:
                self.dataset = StreamingDataset(local=self.local, **kwargs)
                self._temp_dir = None
        except ImportError:
            self.dataset = None
            self._temp_dir = None
    
    def _decompress_shards(self) -> Path:
        """Decompress all shards to a temporary directory."""
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix='mds_decompressed_'))
        
        for shard_info in self.shards:
            basename = shard_info['raw_data']['basename']
            compressed = self.local / (basename + '.bz2')
            uncompressed = temp_dir / basename
            
            if compressed.exists():
                decompress_file(compressed, uncompressed)
        
        return temp_dir
    
    def __len__(self) -> int:
        """Return total number of samples."""
        if self.dataset:
            return len(self.dataset)
        return sum(s.get('samples', 0) for s in self.shards)
    
    def __getitem__(self, idx: int):
        """Get a sample by index."""
        if self.dataset:
            return self.dataset[idx]
        raise NotImplementedError("Direct indexing requires mosaicml-streaming")
    
    def __iter__(self):
        """Iterate over samples."""
        if self.dataset:
            yield from self.dataset
        else:
            # Basic iteration over shard files
            for shard_info in self.shards:
                basename = shard_info['raw_data']['basename']
                shard_path = find_shard_file(self.local, basename)
                with open_compressed(shard_path) as f:
                    yield f.read()
    
    def cleanup(self):
        """Clean up temporary decompressed files if any."""
        if self._temp_dir and self._temp_dir.exists():
            import shutil
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
    
    def __del__(self):
        self.cleanup()


# Convenience function for benchmarks
def load_prolong_dataset(data_path: Optional[Union[str, Path]] = None) -> StreamingDatasetWrapper:
    """
    Load ProLong dataset with automatic compression detection.
    
    Automatically handles both compressed (.mds.bz2) and uncompressed (.mds)
    files based on their existence. Files are decompressed transparently on-the-fly.
    
    Args:
        data_path: Path to dataset (default: auto-detect from data/prolong 
                  or compressed/prolong)
        
    Returns:
        StreamingDatasetWrapper that provides access to the dataset
        
    Example:
        >>> dataset = load_prolong_dataset()
        >>> print(f"Total samples: {len(dataset)}")
        >>> for sample in dataset:
        ...     process(sample)  # Works with .mds or .mds.bz2 transparently
    """
    if data_path is None:
        data_path = get_dataset_path()
    
    return StreamingDatasetWrapper(data_path)
