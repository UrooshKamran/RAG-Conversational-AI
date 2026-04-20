class CartManager:
    """
    Manages the customer's cart state for a single session.
    Tracks items, quantities, prices, applied discounts, and running total.
    """

    PROMOTIONS = {
        "fruits_10_off": {
            "description": "10% OFF all Fruits",
            "applies_to": "Fruits",
            "type": "category_percent",
            "value": 0.10
        },
        "bakery_b2g1": {
            "description": "Buy 2 Get 1 FREE on Bakery",
            "applies_to": "Bakery",
            "type": "b2g1"
        },
        "order_15_off": {
            "description": "15% OFF on orders above $30",
            "applies_to": "order",
            "type": "order_percent",
            "value": 0.15,
            "threshold": 30.0
        },
        "free_delivery": {
            "description": "Free delivery on orders above $25",
            "applies_to": "delivery",
            "type": "free_delivery",
            "threshold": 25.0
        }
    }

    DELIVERY_FEE = 3.00

    def __init__(self):
        self.items = []
        self.applied_promotions = []

    def add_item(self, name, category, quantity, unit_price):
        for item in self.items:
            if item["name"].lower() == name.lower():
                item["quantity"] += quantity
                self._apply_promotions()
                return self.get_summary()
        self.items.append({
            "name": name,
            "category": category,
            "quantity": quantity,
            "unit_price": unit_price
        })
        self._apply_promotions()
        return self.get_summary()

    def remove_item(self, name, quantity=None):
        for i, item in enumerate(self.items):
            if item["name"].lower() == name.lower():
                if quantity is None or quantity >= item["quantity"]:
                    self.items.pop(i)
                else:
                    item["quantity"] -= quantity
                self._apply_promotions()
                return self.get_summary()
        return self.get_summary()

    def clear(self):
        self.items = []
        self.applied_promotions = []

    def _get_subtotal(self):
        return sum(item["unit_price"] * item["quantity"] for item in self.items)

    def _apply_promotions(self):
        self.applied_promotions = []
        subtotal = self._get_subtotal()

        if subtotal > self.PROMOTIONS["order_15_off"]["threshold"]:
            self.applied_promotions.append("order_15_off")

        if subtotal > self.PROMOTIONS["free_delivery"]["threshold"]:
            self.applied_promotions.append("free_delivery")

        categories = {item["category"] for item in self.items}
        if "Fruits" in categories:
            self.applied_promotions.append("fruits_10_off")
        if "Bakery" in categories:
            self.applied_promotions.append("bakery_b2g1")

    def get_discount_amount(self):
        discount = 0.0
        subtotal = self._get_subtotal()

        if "fruits_10_off" in self.applied_promotions:
            fruits_total = sum(
                item["unit_price"] * item["quantity"]
                for item in self.items if item["category"] == "Fruits"
            )
            discount += fruits_total * 0.10

        if "bakery_b2g1" in self.applied_promotions:
            bakery_items = [i for i in self.items if i["category"] == "Bakery"]
            for item in bakery_items:
                free_qty = item["quantity"] // 3
                discount += free_qty * item["unit_price"]

        if "order_15_off" in self.applied_promotions:
            discount += subtotal * 0.15

        return round(discount, 2)

    def get_delivery_fee(self):
        return 0.0 if "free_delivery" in self.applied_promotions else self.DELIVERY_FEE

    def get_total(self):
        subtotal = self._get_subtotal()
        discount = self.get_discount_amount()
        delivery = self.get_delivery_fee()
        return round(subtotal - discount + delivery, 2)

    def get_summary(self):
        subtotal = self._get_subtotal()
        discount = self.get_discount_amount()
        delivery = self.get_delivery_fee()
        total = self.get_total()
        active_promos = [
            self.PROMOTIONS[p]["description"]
            for p in self.applied_promotions
        ]
        return {
            "items": [
                {
                    "name": item["name"],
                    "category": item["category"],
                    "quantity": item["quantity"],
                    "unit_price": item["unit_price"],
                    "line_total": round(item["unit_price"] * item["quantity"], 2)
                }
                for item in self.items
            ],
            "subtotal": round(subtotal, 2),
            "discount": discount,
            "delivery_fee": delivery,
            "total": total,
            "active_promotions": active_promos,
            "item_count": sum(item["quantity"] for item in self.items)
        }

    def to_context_string(self):
        """Serialize cart state into a compact string for LLM prompt injection."""
        if not self.items:
            return "Cart is currently empty."
        summary = self.get_summary()
        lines = ["[CURRENT CART STATE]"]
        for item in summary["items"]:
            lines.append(f"  - {item['name']} x{item['quantity']} @ ${item['unit_price']:.2f} = ${item['line_total']:.2f}")
        lines.append(f"  Subtotal: ${summary['subtotal']:.2f}")
        if summary["discount"] > 0:
            lines.append(f"  Discount: -${summary['discount']:.2f}")
        lines.append(f"  Delivery: ${summary['delivery_fee']:.2f}")
        lines.append(f"  TOTAL: ${summary['total']:.2f}")
        if summary["active_promotions"]:
            lines.append(f"  Promotions applied: {', '.join(summary['active_promotions'])}")
        return "\n".join(lines)
