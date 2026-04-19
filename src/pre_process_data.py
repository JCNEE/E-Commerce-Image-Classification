"""
pre_process_data.py
-------------------
Defines which ImageNet classes count as 'Product' (huntable animals).
No dataset download required.
"""

# List of huntable animals (ImageNet class names)
PRODUCT_CLASSES = [
    "zebra",
    "giraffe",
    "ostrich",
    "antelope",
    "springbok",
    "wildebeest"
]

def get_product_classes():
    """Return the list of product classes."""
    return PRODUCT_CLASSES

if __name__ == "__main__":
    print("Product classes:", get_product_classes())
