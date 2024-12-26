#2. Program on Data Transformation: String Manipulation, Regular Expressions
# Sample text to work with
text = "  Hello, World! Welcome to Python programming.  "

clean_text = text.strip()
print(f"Original Text: '{text}'")
print(f"Text after stripping spaces: '{clean_text}'")

upper_text = clean_text.upper()
print(f"\nText in uppercase: '{upper_text}'")

lower_text = clean_text.lower()
print(f"\nText in lowercase: '{lower_text}'")

count_o = clean_text.count("o")
print(f"\nNumber of occurrences of 'o': {count_o}")

replaced_text = clean_text.replace("Python", "Data Science")
print(f"\nText after replacing 'Python' with 'Data Science': '{replaced_text}'")

position_world = clean_text.find("World")
print(f"\nPosition of 'World' in the text: {position_world}")

words = clean_text.split()
print(f"\nList of words in the text: {words}")

joined_text = " ".join(words)
print(f"\nText after joining words: '{joined_text}'")

starts_with_hello = clean_text.startswith("Hello")
print(f"\nDoes the text start with 'Hello'? {starts_with_hello}")

ends_with_programming = clean_text.endswith("programming.")
print(f"\nDoes the text end with 'programming.'? {ends_with_programming}")

import re

text = """
John's email is [email protected]. He said, "Python is awesome!!"    It's a     great   language.
Another email: [email protected]. 
"""

clean_text = re.sub(r"[^a-zA-Z0-9@\.\s]", "", text)
print("Text after removing special characters:")
print(clean_text)

clean_text = clean_text.lower()
print("\nText after converting to lowercase:")
print(clean_text)

clean_text = re.sub(r"\s+", " ", clean_text)
print("\nText after replacing multiple spaces:")
print(clean_text)

vowel_words = re.findall(r"\b[aeiouAEIOU]\w+", clean_text)
print("\nWords starting with a vowel:")
print(vowel_words)

masked_text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[email protected]", clean_text)
print("\nText after replacing emails:")
print(masked_text)
