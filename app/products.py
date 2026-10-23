import csv


def load_products(path):
    """Baca products.csv. Hasilnya dict: class_name -> {"name": ..., "price": ...}."""
    products = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            products[row["class_name"]] = {
                "name": row["display_name"],
                "price": int(row["price"]),
            }
    return products
