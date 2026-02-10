#!/usr/bin/env python3
"""
Test PageRank implementation on a known small graph
Independent of randomly generated files
"""

from collections import defaultdict
import statistics

def compute_pagerank_test(graph, pages, damping=0.85, tolerance=0.005, max_iterations=100):
    """PageRank algorithm with dangling node handling"""
    n = len(pages)
    pagerank = {page: 1.0/n for page in pages}
    
    # Build incoming links map
    incoming = defaultdict(list)
    for page in pages:
        for target in graph[page]:
            incoming[target].append(page)
    
    # Identify dangling nodes (nodes with no outgoing links)
    dangling_nodes = [page for page in pages if len(graph[page]) == 0]
    
    iteration = 0
    while iteration < max_iterations:
        new_pagerank = {}
        
        # Calculate dangling contribution (distribute evenly to all pages)
        dangling_sum = sum(pagerank[page] for page in dangling_nodes)
        
        for page in pages:
            # Base probability + dangling contribution
            rank = (1 - damping) / n + damping * (dangling_sum / n)
            
            # Add contributions from incoming links
            for incoming_page in incoming[page]:
                outgoing_count = len(graph[incoming_page])
                if outgoing_count > 0:
                    rank += damping * (pagerank[incoming_page] / outgoing_count)
            
            new_pagerank[page] = rank
        
        # Check convergence
        total_old = sum(pagerank.values())
        total_new = sum(new_pagerank.values())
        change = abs(total_new - total_old) / total_old
        
        pagerank = new_pagerank
        iteration += 1
        
        if change < tolerance:
            break
    
    return pagerank

def test_simple_graph():
    """
    Test on a simple 4-node graph:
    A -> B, C
    B -> C
    C -> A
    D -> C
    """
    print("Test 1: Simple 4-node graph")
    print("-" * 40)
    
    graph = {
        'A': ['B', 'C'],
        'B': ['C'],
        'C': ['A'],
        'D': ['C']
    }
    pages = {'A', 'B', 'C', 'D'}
    
    pagerank = compute_pagerank_test(graph, pages)
    
    # Check sum equals 1
    total = sum(pagerank.values())
    print(f"Sum of PageRanks: {total:.6f}")
    assert abs(total - 1.0) < 0.001, f"Sum should be 1.0, got {total}"
    
    print("PageRank values:")
    for page in sorted(pagerank.keys()):
        print(f"  {page}: {pagerank[page]:.6f}")
    
    # C should have highest PageRank (receives most links)
    max_page = max(pagerank, key=pagerank.get)
    print(f"Highest PageRank: {max_page}")
    assert max_page == 'C', f"Expected C to have highest PageRank, got {max_page}"
    
    print("✓ Test 1 PASSED\n")

def test_linear_chain():
    """
    Test on linear chain with dangling node: A -> B -> C -> D (D has no outgoing links)
    """
    print("Test 2: Linear chain graph (with dangling node)")
    print("-" * 40)
    
    graph = {
        'A': ['B'],
        'B': ['C'],
        'C': ['D'],
        'D': []  # Dangling node
    }
    pages = {'A', 'B', 'C', 'D'}
    
    pagerank = compute_pagerank_test(graph, pages)
    
    total = sum(pagerank.values())
    print(f"Sum of PageRanks: {total:.6f}")
    assert abs(total - 1.0) < 0.001, f"Sum should be 1.0, got {total}"
    
    print("PageRank values:")
    for page in sorted(pagerank.keys()):
        print(f"  {page}: {pagerank[page]:.6f}")
    
    print("✓ Test 2 PASSED\n")

def test_complete_graph():
    """
    Test on complete graph where everyone links to everyone
    """
    print("Test 3: Complete graph (all-to-all)")
    print("-" * 40)
    
    nodes = ['A', 'B', 'C', 'D', 'E']
    graph = {node: [n for n in nodes if n != node] for node in nodes}
    pages = set(nodes)
    
    pagerank = compute_pagerank_test(graph, pages)
    
    total = sum(pagerank.values())
    print(f"Sum of PageRanks: {total:.6f}")
    assert abs(total - 1.0) < 0.001, f"Sum should be 1.0, got {total}"
    
    # All should have equal PageRank
    values = list(pagerank.values())
    std_dev = statistics.stdev(values)
    print(f"Standard deviation of PageRanks: {std_dev:.6f}")
    assert std_dev < 0.001, "All nodes should have equal PageRank in complete graph"
    
    print("PageRank values:")
    for page in sorted(pagerank.keys()):
        print(f"  {page}: {pagerank[page]:.6f}")
    
    print("✓ Test 3 PASSED\n")

def test_all_dangling():
    """
    Test where all nodes are dangling (no outgoing links)
    Should converge to uniform distribution
    """
    print("Test 4: All dangling nodes")
    print("-" * 40)
    
    graph = {
        'A': [],
        'B': [],
        'C': []
    }
    pages = {'A', 'B', 'C'}
    
    pagerank = compute_pagerank_test(graph, pages)
    
    total = sum(pagerank.values())
    print(f"Sum of PageRanks: {total:.6f}")
    assert abs(total - 1.0) < 0.001, f"Sum should be 1.0, got {total}"
    
    print("PageRank values:")
    for page in sorted(pagerank.keys()):
        print(f"  {page}: {pagerank[page]:.6f}")
    
    # All should be equal (uniform distribution)
    values = list(pagerank.values())
    for val in values:
        assert abs(val - 1.0/3) < 0.001, "All dangling nodes should have equal PageRank"
    
    print("✓ Test 4 PASSED\n")

def main():
    print("="*50)
    print("PageRank Correctness Tests")
    print("="*50)
    print()
    
    test_simple_graph()
    test_linear_chain()
    test_complete_graph()
    test_all_dangling()
    
    print("="*50)
    print("All tests PASSED! ✓")
    print("="*50)

if __name__ == "__main__":
    main()
