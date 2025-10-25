"""
Chapter 01 Snippet: File Analyzer

Demonstrates:
- File I/O operations
- Context managers (with statement)
- String processing
- Exception handling
"""

from typing import Dict
import os


def analyze_file(filename: str) -> Dict[str, int]:
    """
    Analyze a text file and return statistics.
    
    Returns dictionary with:
    - lines: number of lines
    - words: number of words
    - chars: number of characters
    - unique_words: number of unique words
    
    Laravel equivalent: Similar to File::get() + str functions
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    stats = {
        "lines": 0,
        "words": 0,
        "chars": 0,
        "unique_words": 0
    }
    
    unique_words = set()
    
    # CONCEPT: Context Manager (with statement)
    # Automatically closes file after block
    # Like Laravel's try-finally for cleanup
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            stats["lines"] += 1
            stats["chars"] += len(line)
            
            # Split line into words
            words = line.split()
            stats["words"] += len(words)
            
            # Track unique words (case-insensitive)
            for word in words:
                # Clean word: remove punctuation
                clean_word = ''.join(c for c in word if c.isalnum()).lower()
                if clean_word:
                    unique_words.add(clean_word)
    
    stats["unique_words"] = len(unique_words)
    
    return stats


def write_stats_file(stats: Dict[str, int], output_file: str) -> None:
    """
    Write statistics to a file.
    
    Demonstrates: Writing text files
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("File Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        
        for key, value in stats.items():
            f.write(f"{key.replace('_', ' ').title()}: {value:,}\n")
        
        f.write("\n" + "=" * 40 + "\n")


def create_sample_file(filename: str = "sample.txt") -> None:
    """Create a sample text file for testing."""
    sample_text = """Python is a high-level programming language.
It emphasizes code readability and simplicity.
Python is widely used for web development, data analysis, and automation.
The language has a large standard library and active community."""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print(f"Created sample file: {filename}")


def main():
    """Demo the file analyzer."""
    print("File Analyzer Demo")
    print("=" * 60)
    
    # Create sample file
    sample_file = "sample_analysis.txt"
    create_sample_file(sample_file)
    
    try:
        # Analyze the file
        print(f"\nAnalyzing {sample_file}...")
        stats = analyze_file(sample_file)
        
        # Display results
        print("\nResults:")
        print("-" * 40)
        for key, value in stats.items():
            label = key.replace('_', ' ').title()
            print(f"{label:.<30} {value:>8,}")
        
        # Calculate additional metrics
        if stats["words"] > 0:
            avg_word_length = stats["chars"] / stats["words"]
            print(f"{'Avg Word Length':.<30} {avg_word_length:>8.2f}")
        
        # Write to output file
        output_file = "analysis_report.txt"
        write_stats_file(stats, output_file)
        print(f"\n✓ Report saved to {output_file}")
        
        # Cleanup
        os.remove(sample_file)
        os.remove(output_file)
        print(f"✓ Cleaned up temporary files")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

