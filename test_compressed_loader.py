#!/usr/bin/env python3
"""
Test suite for compressed dataset loader.

Tests transparent access to .mds files that may be compressed (.mds.bz2).
"""

import bz2
import os
import tempfile
import unittest
from pathlib import Path

from data.compressed_dataset_loader import (
    CompressedFileHandle,
    open_compressed,
    find_shard_file,
    list_shard_files,
    decompress_file,
)


class TestCompressedFileHandle(unittest.TestCase):
    """Test the CompressedFileHandle class."""
    
    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_content = b"Hello, this is test content for compression!" * 100
        
        # Create uncompressed file
        self.uncompressed_path = Path(self.temp_dir) / "test.mds"
        with open(self.uncompressed_path, 'wb') as f:
            f.write(self.test_content)
        
        # Create compressed file
        self.compressed_path = Path(self.temp_dir) / "test_compressed.mds.bz2"
        with bz2.open(self.compressed_path, 'wb') as f:
            f.write(self.test_content)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_read_uncompressed(self):
        """Test reading uncompressed file."""
        handle = CompressedFileHandle(self.uncompressed_path)
        content = handle.read()
        handle.close()
        self.assertEqual(content, self.test_content)
        self.assertFalse(handle.is_compressed)
    
    def test_read_compressed(self):
        """Test reading compressed file."""
        handle = CompressedFileHandle(self.compressed_path)
        content = handle.read()
        handle.close()
        self.assertEqual(content, self.test_content)
        self.assertTrue(handle.is_compressed)
    
    def test_context_manager(self):
        """Test using as context manager."""
        with CompressedFileHandle(self.uncompressed_path) as f:
            content = f.read()
        self.assertEqual(content, self.test_content)
    
    def test_seek_and_tell(self):
        """Test seek and tell operations."""
        with CompressedFileHandle(self.uncompressed_path) as f:
            f.seek(10)
            self.assertEqual(f.tell(), 10)
            data = f.read(5)
            self.assertEqual(f.tell(), 15)
            self.assertEqual(data, self.test_content[10:15])


class TestOpenCompressed(unittest.TestCase):
    """Test the open_compressed convenience function."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_content = b"Test content for open_compressed"
        
        # Create compressed file
        self.compressed_path = Path(self.temp_dir) / "test.mds.bz2"
        with bz2.open(self.compressed_path, 'wb') as f:
            f.write(self.test_content)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_open_uncompressed_name_uses_compressed(self):
        """Test that .mds name finds .mds.bz2 if .mds doesn't exist."""
        # Request .mds but only .mds.bz2 exists
        with open_compressed(Path(self.temp_dir) / "test.mds") as f:
            content = f.read()
        self.assertEqual(content, self.test_content)


class TestFindShardFile(unittest.TestCase):
    """Test finding shard files."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create uncompressed shard
        self.uncompressed = Path(self.temp_dir) / "shard.00000.mds"
        self.uncompressed.write_bytes(b"uncompressed data")
        
        # Create compressed shard
        self.compressed = Path(self.temp_dir) / "shard.00001.mds.bz2"
        with bz2.open(self.compressed, 'wb') as f:
            f.write(b"compressed data")
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_find_uncompressed(self):
        """Test finding uncompressed shard."""
        path = find_shard_file(self.temp_dir, "shard.00000.mds")
        self.assertEqual(path, self.uncompressed)
    
    def test_find_compressed(self):
        """Test finding compressed shard when uncompressed doesn't exist."""
        path = find_shard_file(self.temp_dir, "shard.00001.mds")
        self.assertEqual(path, self.compressed)
    
    def test_prefer_uncompressed(self):
        """Test that uncompressed is preferred when both exist."""
        # Create both versions
        both_compressed = Path(self.temp_dir) / "shard.00002.mds.bz2"
        both_uncompressed = Path(self.temp_dir) / "shard.00002.mds"
        with bz2.open(both_compressed, 'wb') as f:
            f.write(b"compressed")
        both_uncompressed.write_bytes(b"uncompressed")
        
        path = find_shard_file(self.temp_dir, "shard.00002.mds")
        self.assertEqual(path, both_uncompressed)
    
    def test_not_found(self):
        """Test FileNotFoundError when neither exists."""
        with self.assertRaises(FileNotFoundError):
            find_shard_file(self.temp_dir, "nonexistent.mds")


class TestListShardFiles(unittest.TestCase):
    """Test listing shard files."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mix of files
        (Path(self.temp_dir) / "shard.00000.mds").write_bytes(b"data")
        with bz2.open(Path(self.temp_dir) / "shard.00001.mds.bz2", 'wb') as f:
            f.write(b"compressed")
        # Both versions
        with bz2.open(Path(self.temp_dir) / "shard.00002.mds.bz2", 'wb') as f:
            f.write(b"compressed")
        (Path(self.temp_dir) / "shard.00002.mds").write_bytes(b"uncompressed")
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_list_shards(self):
        """Test listing all shards."""
        shards = list_shard_files(self.temp_dir)
        # Should have 3 entries (00000, 00001, 00002)
        self.assertEqual(len(shards), 3)
        
        # Check sorting
        names = [s[0] for s in shards]
        self.assertEqual(names, sorted(names))
        
        # Check compression flags
        compression_flags = {s[0]: s[1] for s in shards}
        self.assertFalse(compression_flags["shard.00000.mds"])  # uncompressed
        self.assertTrue(compression_flags["shard.00001.mds"])   # only compressed
        self.assertFalse(compression_flags["shard.00002.mds"])  # both, prefer uncompressed


class TestDecompressFile(unittest.TestCase):
    """Test decompressing files."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_content = b"Content to decompress"
        
        self.compressed_path = Path(self.temp_dir) / "compressed.mds.bz2"
        with bz2.open(self.compressed_path, 'wb') as f:
            f.write(self.test_content)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_decompress(self):
        """Test basic decompression."""
        output_path = Path(self.temp_dir) / "output.mds"
        result = decompress_file(self.compressed_path, output_path)
        
        self.assertEqual(result, output_path)
        self.assertEqual(output_path.read_bytes(), self.test_content)
        # Original should still exist
        self.assertTrue(self.compressed_path.exists())
    
    def test_decompress_auto_name(self):
        """Test decompression with automatic output name."""
        result = decompress_file(self.compressed_path)
        
        expected = Path(self.temp_dir) / "compressed.mds"
        self.assertEqual(result, expected)
        self.assertEqual(expected.read_bytes(), self.test_content)
    
    def test_decompress_and_remove(self):
        """Test decompression with original removal."""
        output_path = Path(self.temp_dir) / "output.mds"
        decompress_file(self.compressed_path, output_path, remove_original=True)
        
        self.assertFalse(self.compressed_path.exists())
        self.assertEqual(output_path.read_bytes(), self.test_content)


class TestRealDataset(unittest.TestCase):
    """Test with real compressed dataset if available."""
    
    def test_compressed_prolong_exists(self):
        """Check if compressed ProLong dataset exists."""
        compressed_path = Path("compressed/prolong")
        if not compressed_path.exists():
            self.skipTest("Compressed ProLong dataset not found")
        
        # Should have some .mds.bz2 files
        bz2_files = list(compressed_path.rglob("*.mds.bz2"))
        self.assertGreater(len(bz2_files), 0, "No .mds.bz2 files found")
        
        print(f"\nFound {len(bz2_files)} compressed .mds.bz2 files")
    
    def test_read_compressed_prolong_shard(self):
        """Test reading a compressed ProLong shard."""
        compressed_path = Path("compressed/prolong")
        if not compressed_path.exists():
            self.skipTest("Compressed ProLong dataset not found")
        
        bz2_files = list(compressed_path.rglob("*.mds.bz2"))
        if not bz2_files:
            self.skipTest("No compressed shards found")
        
        # Try to read first compressed file
        with open_compressed(bz2_files[0]) as f:
            header = f.read(100)
        
        # Should be able to read without errors
        self.assertIsInstance(header, bytes)
        self.assertGreater(len(header), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
