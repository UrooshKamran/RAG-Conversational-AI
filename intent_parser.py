"""
intent_parser.py
Lightweight keyword-based intent parser.
Detects add/remove cart actions from user messages and updates CartManager.
No ML — pure string matching against the known product catalog.
"""

import re

# Full catalog mirroring system_prompt.py
CATALOG = {
    "Apple":             {"category": "Fruits",     "price": 2.50},
    "Banana":            {"category": "Fruits",     "price": 1.80},
    "Mango":             {"category": "Fruits",     "price": 3.50},
    "Strawberry":        {"category": "Fruits",     "price": 2.20},
    "Watermelon":        {"category": "Fruits",     "price": 5.00},
    "Grapes":            {"category": "Fruits",     "price": 2.80},
    "Orange":            {"category": "Fruits",     "price": 2.00},
    "Pineapple":         {"category": "Fruits",     "price": 3.00},
    "Tomato":            {"category": "Vegetables", "price": 1.50},
    "Potato":            {"category": "Vegetables", "price": 2.00},
    "Onion":             {"category": "Vegetables", "price": 1.20},
    "Spinach":           {"category": "Vegetables", "price": 1.00},
    "Carrot":            {"category": "Vegetables", "price": 1.80},
    "Broccoli":          {"category": "Vegetables", "price": 2.50},
    "Cucumber":          {"category": "Vegetables", "price": 0.80},
    "Bell Pepper":       {"category": "Vegetables", "price": 2.20},
    "Full Cream Milk":   {"category": "Dairy",      "price": 1.80},
    "Skimmed Milk":      {"category": "Dairy",      "price": 1.70},
    "Cheddar Cheese":    {"category": "Dairy",      "price": 3.20},
    "Mozzarella Cheese": {"category": "Dairy",      "price": 3.50},
    "Greek Yogurt":      {"category": "Dairy",      "price": 2.80},
    "Butter":            {"category": "Dairy",      "price": 2.50},
    "Cream Cheese":      {"category": "Dairy",      "price": 2.20},
    "Sour Cream":        {"category": "Dairy",      "price": 1.90},
    "White Bread":       {"category": "Bakery",     "price": 1.50},
    "Whole Wheat Bread": {"category": "Bakery",     "price": 2.00},
    "Croissant":         {"category": "Bakery",     "price": 1.20},
    "Sourdough Bread":   {"category": "Bakery",     "price": 3.50},
    "Bagel":             {"category": "Bakery",     "price": 2.80},
    "Dinner Rolls":      {"category": "Bakery",     "price": 2.20},
    "Blueberry Muffin":  {"category": "Bakery",     "price": 1.80},
    "Cinnamon Roll":     {"category": "Bakery",     "price": 2.00},
    "Mineral Water":     {"category": "Beverages",  "price": 0.90},
    "Orange Juice":      {"category": "Beverages",  "price": 2.50},
    "Apple Juice":       {"category": "Beverages",  "price": 2.20},
    "Green Tea":         {"category": "Beverages",  "price": 3.00},
    "Coffee":            {"category": "Beverages",  "price": 5.50},
    "Coca-Cola":         {"category": "Beverages",  "price": 1.80},
    "Sparkling Water":   {"category": "Beverages",  "price": 1.20},
    "Almond Milk":       {"category": "Beverages",  "price": 3.20},
    "Salted Chips":      {"category": "Snacks",     "price": 1.50},
    "Mixed Nuts":        {"category": "Snacks",     "price": 4.50},
    "Dark Chocolate Bar":{"category": "Snacks",     "price": 2.80},
    "Granola Bar":       {"category": "Snacks",     "price": 3.20},
    "Popcorn":           {"category": "Snacks",     "price": 2.50},
    "Rice Crackers":     {"category": "Snacks",     "price": 1.80},
    "Gummy Bears":       {"category": "Snacks",     "price": 2.00},
    "Pretzels":          {"category": "Snacks",     "price": 2.20},
}

ADD_KEYWORDS    = ["add", "put", "include", "want", "get", "order", "i'll take", "i will take"]
REMOVE_KEYWORDS = ["remove", "take out", "delete", "cancel", "drop", "don't want", "do not want"]


def _extract_quantity(text):
    """Extract a numeric quantity from text. Returns 1 if none found."""
    match = re.search(r'\b(\d+)\b', text)
    return int(match.group(1)) if match else 1


def _find_product(text):
    """Find the first matching product name in text (case-insensitive)."""
    text_lower = text.lower()
    # Sort by length descending so multi-word names match before single words
    for name in sorted(CATALOG.keys(), key=len, reverse=True):
        variants = [name.lower(), name.lower() + "s", name.lower().rstrip("y") + "ies"]
        if any(v in text_lower for v in variants):
            return name
    return None


def parse_intent(user_message, cart):
    """Parse ALL add/remove intents from a single message."""
    msg_lower = user_message.lower()
    results = []

    is_add    = any(kw in msg_lower for kw in ADD_KEYWORDS)
    is_remove = any(kw in msg_lower for kw in REMOVE_KEYWORDS)

    if not is_add and not is_remove:
        return {"action": None, "product": None, "quantity": 1, "cart_summary": cart.get_summary()}

    # Split message by "and" to handle multiple items
    parts = re.split(r'\band\b|,', user_message, flags=re.IGNORECASE)

    for part in parts:
        part = part.strip()
        product_name = _find_product(part)
        if not product_name:
            continue
        quantity = _extract_quantity(part)
        product  = CATALOG[product_name]

        if is_remove:
            cart.remove_item(product_name, quantity if quantity > 1 else None)
            results.append({"action": "remove", "product": product_name, "quantity": quantity})
        elif is_add:
            cart.add_item(product_name, product["category"], quantity, product["price"])
            results.append({"action": "add", "product": product_name, "quantity": quantity})

    return {
        "action": results[0]["action"] if results else None,
        "product": results[0]["product"] if results else None,
        "quantity": results[0]["quantity"] if results else 1,
        "cart_summary": cart.get_summary()
    }