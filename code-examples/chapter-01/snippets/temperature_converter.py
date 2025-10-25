"""
Chapter 01 Snippet: Temperature Converter

Demonstrates:
- Functions with type hints
- Return values
- Mathematical operations
- F-strings for formatting
"""


def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Convert Celsius to Fahrenheit.
    
    Formula: (C × 9/5) + 32 = F
    
    Args:
        celsius: Temperature in Celsius
    
    Returns:
        Temperature in Fahrenheit
    """
    return (celsius * 9/5) + 32


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """
    Convert Fahrenheit to Celsius.
    
    Formula: (F - 32) × 5/9 = C
    
    Args:
        fahrenheit: Temperature in Fahrenheit
    
    Returns:
        Temperature in Celsius
    """
    return (fahrenheit - 32) * 5/9


def main():
    """Demo the temperature conversions."""
    # Test conversions
    print("Temperature Converter")
    print("=" * 40)
    
    # Celsius to Fahrenheit
    temps_c = [0, 10, 20, 30, 100]
    print("\nCelsius to Fahrenheit:")
    for temp in temps_c:
        result = celsius_to_fahrenheit(temp)
        print(f"  {temp:>5}°C = {result:>7.1f}°F")
    
    # Fahrenheit to Celsius
    temps_f = [32, 50, 68, 86, 212]
    print("\nFahrenheit to Celsius:")
    for temp in temps_f:
        result = fahrenheit_to_celsius(temp)
        print(f"  {temp:>5}°F = {result:>7.1f}°C")


if __name__ == "__main__":
    main()

