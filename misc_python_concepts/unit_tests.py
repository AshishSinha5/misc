
import unittest

# sample class 
class Book:
    def __init__(self, title, author, year, price, pages, discount):
        self.title = title
        self.author = author
        self.year = year
        self.price = price
        self.pages = pages
        self.discount = discount

    def __str__(self):
        return f"{self.title} by {self.author} in {self.year}"
    

    def get_reading_time(self):
        return f"{self.pages* 1.5} minutes"
    
    def apply_discount(self):
        discounted_price = self.price - (self.price * self.discount)
        return f"${discounted_price}"
    

class TestBook(unittest.TestCase):
    def test_get_reading_time(self):
        book1 = Book("The Great Gatsby", "F. Scott Fitzgerald", 1925, 15.00, 180, 0.10)
        book2 = Book("The Catcher in the Rye", "J.D. Salinger", 1951, 12.00, 240, 0.15)
        self.assertEqual(book1.get_reading_time(), "270.0 minutes")
        self.assertEqual(book2.get_reading_time(), "360.0 minutes")

    def test_apply_discount(self):
        book1 = Book("The Great Gatsby", "F. Scott Fitzgerald", 1925, 15.00, 180, 0.10)
        book2 = Book("The Catcher in the Rye", "J.D. Salinger", 1951, 12.00, 240, 0.15)
        self.assertEqual(book1.apply_discount(), "$13.5")
        self.assertEqual(book2.apply_discount(), "$10.2")

if __name__ == "__main__":
    unittest.main()

    