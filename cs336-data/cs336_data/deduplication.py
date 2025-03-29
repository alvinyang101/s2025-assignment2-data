import os
import hashlib
import random
import re
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Set, Dict, Tuple, Union


def exact_line_deduplication(
    input_files: List[Union[str, os.PathLike]], 
    output_directory: Union[str, os.PathLike]
):
    """
    Args:
        input_files: A list of paths to input files
        output_directory: Path to the output directory
    """
    # Convert input paths to Path objects
    input_paths = [Path(file) for file in input_files]
    output_dir = Path(output_directory)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # First pass: Count frequency of each line across all files
    line_counter = Counter()
    
    for file_path in input_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # Use hash to reduce memory usage for very large files
                    line_hash = hashlib.md5(line.encode()).hexdigest()
                    line_counter[line_hash] += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Second pass: Rewrite each file with only unique lines
    for file_path in input_paths:
        try:
            # Get output file path with same name in output directory
            output_path = output_dir / file_path.name
            
            with open(file_path, 'r', encoding='utf-8') as infile, \
                 open(output_path, 'w', encoding='utf-8') as outfile:
                
                # Set to keep track of lines already written to this file
                # (handles duplicates within the same file)
                written_lines = set()
                
                for line in infile:
                    line_hash = hashlib.md5(line.encode()).hexdigest()
                    
                    # Only keep the line if it appears exactly once in the corpus
                    if line_counter[line_hash] == 1 and line_hash not in written_lines:
                        outfile.write(line)
                        written_lines.add(line_hash)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def minhash_deduplication(
    input_files: List[Union[str, os.PathLike]],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: Union[str, os.PathLike],
):
    """
    Write a function that takes a list of paths to input files and performs fuzzy document deduplication
    with minhash and LSH. In particular, your function should compute minhash signatures for each
    document in the provided list of paths, use LSH with the provided number of bands to identify candidate
    duplicates, and then compute the true ngram Jaccard similarity between candidate duplicates and
    remove those that exceed a given threshold. To improve recall (following Penedo et al., 2023), normalize
    the text before computing minhash signatures and/or comparing Jaccard similarity by lowercasing,
    removing punctuation, normalizing whitespaces, and removing accents, and applying NFD unicode
    normalization.
    
    Args:
        input_files: List of paths to input files
        num_hashes: Number of hash functions for MinHash signatures
        num_bands: Number of bands for LSH
        ngrams: N-gram length (in words) for computing MinHash signatures
        jaccard_threshold: Threshold above which documents are considered duplicates
        output_directory: Path to write deduplicated files
    
    Returns:
        None
    """    
    input_paths = [Path(path) for path in input_files]
    output_dir = Path(output_directory)
    os.makedirs(output_dir, exist_ok=True)
    
    documents = {}
    for path in input_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents[str(path)] = normalize_text(content)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    
    if not documents:
        print("No documents were loaded.")
        return
    
    # Generate ngram sets for each document
    ngram_sets = {
        doc_id: generate_ngrams(text, ngrams)
        for doc_id, text in documents.items()
    }
    
    # Compute MinHash signatures
    signatures = compute_minhash_signatures(ngram_sets, num_hashes)
    
    # LSH to find candidate duplicates
    candidates = find_candidate_duplicates(signatures, num_bands)
    
    # Compute actual Jaccard similarities for candidates
    similar_docs = compute_jaccard_similarities(
        candidates, ngram_sets, jaccard_threshold
    )
    clusters = cluster_similar_documents(similar_docs)
    
    # Select documents to keep
    docs_to_keep = select_documents_to_keep(clusters, documents.keys())
    
    # Write deduplicated documents to output directory
    for doc_id in docs_to_keep:
        input_path = Path(doc_id)
        output_path = output_dir / input_path.name
        
        try:
            with open(input_path, 'r', encoding='utf-8') as infile, \
                 open(output_path, 'w', encoding='utf-8') as outfile:
                outfile.write(infile.read())
        except Exception as e:
            print(f"Error writing {output_path}: {e}")
    
    print(f"Kept {len(docs_to_keep)} out of {len(documents)} documents.")


def normalize_text(text: str) -> str:
    """
    Normalize the text before computing minhash signatures and/or comparing Jaccard similarity by lowercasing,
    removing punctuation, normalizing whitespaces, and removing accents, and applying NFD unicode
    normalization.
    Args:
        text: Input text to normalize
    
    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()
    
    # Punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove accents, apply NFD unicode normalization
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if not unicodedata.combining(c)
    )
    
    return text


def generate_ngrams(text: str, n: int) -> Set[str]:
    """
    Args:
        text: Input text
        n: Size of n-grams (in words)
    
    Returns:
        Set of n-gram 
    """
    words = text.split()
    
    if len(words) < n:
        return {' '.join(words)}
    
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    
    return ngrams


def compute_minhash_signatures(
    ngram_sets: Dict[str, Set[str]], 
    num_hashes: int
) -> Dict[str, List[int]]:
    """    
    Args:
        ngram_sets: Dictionary mapping document IDs to their ngram sets
        num_hashes: Number of hash functions to use
    
    Returns:
        Dictionary mapping document IDs to their MinHash signatures
    """
    all_ngrams = set()
    for ngrams in ngram_sets.values():
        all_ngrams.update(ngrams)
    
    ngram_to_int = {ngram: i for i, ngram in enumerate(all_ngrams)}
    
    # Create hash functions
    hash_functions = []
    random.seed(1)
    for i in range(num_hashes):
        a = random.randint(1, 2**31 - 1)
        b = random.randint(0, 2**31 - 1)
        p = 2**31 - 1
        
        def hash_func(x, a=a, b=b, p=p):
            return (a * x + b) % p
        
        hash_functions.append(hash_func)
    
    # Compute MinHash signatures
    signatures = {}
    for doc_id, ngrams in ngram_sets.items():
        doc_ngram_ints = {ngram_to_int[s] for s in ngrams if s in ngram_to_int}
        
        if not doc_ngram_ints:
            signatures[doc_id] = [0] * num_hashes
            continue
        
        signature = [float('inf')] * num_hashes
        
        # Compute minimum hash values
        for ngram_int in doc_ngram_ints:
            for i, hash_func in enumerate(hash_functions):
                hash_value = hash_func(ngram_int)
                signature[i] = min(signature[i], hash_value)
        
        signatures[doc_id] = signature
    
    return signatures


def find_candidate_duplicates(
    signatures: Dict[str, List[int]], 
    num_bands: int
) -> Set[Tuple[str, str]]:
    """    
    Args:
        signatures: Dictionary mapping document IDs to their MinHash signatures
        num_bands: Number of bands for LSH
    
    Returns:
        Set of tuples representing candidate duplicate pairs
    """
    rows_per_band = len(next(iter(signatures.values()))) // num_bands
    
    buckets = [defaultdict(list) for _ in range(num_bands)]
    
    # Hash signature bands into buckets
    for doc_id, signature in signatures.items():
        for band in range(num_bands):
            band_signature = tuple(
                signature[band * rows_per_band : (band + 1) * rows_per_band]
            )
            
            band_hash = hash(band_signature)
            buckets[band][band_hash].append(doc_id)
    
    # Find candidate pairs
    candidates = set()
    for band_buckets in buckets:
        for bucket in band_buckets.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        pair = tuple(sorted([bucket[i], bucket[j]]))
                        candidates.add(pair)
    
    return candidates


def compute_jaccard_similarities(
    candidates: Set[Tuple[str, str]],
    ngram_sets: Dict[str, Set[str]],
    threshold: float
) -> Dict[Tuple[str, str], float]:
    """
    Args:
        candidates: Set of candidate document pairs
        ngram_sets: Dictionary mapping document IDs to their ngram sets
        threshold: Similarity threshold to consider documents as duplicates
    
    Returns:
        Dictionary mapping document pairs to their Jaccard similarity
    """
    similar_docs = {}
    
    for doc1, doc2 in candidates:
        ngrams1 = ngram_sets[doc1]
        ngrams2 = ngram_sets[doc2]
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        similarity = intersection / union if union > 0 else 0
        
        if similarity >= threshold:
            similar_docs[(doc1, doc2)] = similarity
    
    return similar_docs


def cluster_similar_documents(documents):
    """    
    Args:
        similar_docs: Dictionary of document pairs and their similarities
    
    Returns:
        List of document clusters
    """
    graph = defaultdict(set)
    for (doc1, doc2) in documents:
        graph[doc1].add(doc2)
        graph[doc2].add(doc1)
    
    clusters = []
    visited = set()
    
    for doc in graph:
        if doc not in visited:
            cluster = set()
            to_visit = [doc]
            visited.add(doc)
            
            # Find all connected documents
            while to_visit:
                current = to_visit.pop(0)
                cluster.add(current)
                
                # Add unvisited neighbors to the queue
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        to_visit.append(neighbor)
            
            clusters.append(cluster)
    
    return clusters


def select_documents_to_keep(
    clusters: List[Set[str]],
    all_docs: Set[str]
) -> Set[str]:
    """
    Args:
        clusters: List of document clusters
        all_docs: Set of all document IDs
    
    Returns:
        Set of document IDs to keep
    """
    docs_to_keep = set()
    
    # Documents that are not part of any cluster
    lonely_docs = set(all_docs) - set().union(*clusters) if clusters else set(all_docs)
    docs_to_keep.update(lonely_docs)
    
    # Select one document to keep from each cluster
    for cluster in clusters:
        keeper = random.choice(list(cluster))
        docs_to_keep.add(keeper)
    
    return docs_to_keep