"""
Phase III - Multi-Turn Dialogue Tests
Tests the ConversationManager with 4 scripted dialogues covering:
1. Browsing and adding items
2. Applying promotions and checkout
3. Out-of-stock / substitution handling
4. Policy inquiry
"""

import sys
import time
from conversation_manager import ConversationManager


def print_separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def run_dialogue(dialogue_name, turns):
    print_separator(dialogue_name)
    cm = ConversationManager(session_id=dialogue_name.replace(" ", "_").lower())

    for user_msg in turns:
        print(f"\nCustomer: {user_msg}")
        print("Shabo:  ", end="", flush=True)

        start = time.time()
        response = cm.chat(user_msg)
        elapsed = time.time() - start

        print(response)
        print(f"  [{elapsed:.2f}s | turns={cm.turn_count} | active_msgs={cm.memory.get_turn_count()}]")

        if not cm.is_active:
            print("\n  [Session ended]")
            break

    print("\n  --- Cart State ---")
    cart = cm.cart.get_summary()
    if cart["items"]:
        for item in cart["items"]:
            print(f"  {item['name']} x{item['quantity']} = ${item['line_total']:.2f}")
        print(f"  Subtotal: ${cart['subtotal']:.2f}")
        if cart["discount"] > 0:
            print(f"  Discount: -${cart['discount']:.2f}")
        print(f"  Delivery: ${cart['delivery_fee']:.2f}")
        print(f"  Total: ${cart['total']:.2f}")
        if cart["active_promotions"]:
            print(f"  Promos: {', '.join(cart['active_promotions'])}")
    else:
        print("  (empty)")

    print("\n  --- Session State ---")
    state = cm.get_session_state()
    print(f"  Turns: {state['turn_count']} | Summary: {state['has_summary']} | Active: {state['is_active']}")


DIALOGUE_1 = [
    "Hi there! What categories do you have?",
    "What fruits do you have and how much are they?",
    "Add 2 kg of apples and 1 dozen bananas to my cart please.",
    "Also add 1 litre of full cream milk.",
    "What is my current total?"
]

DIALOGUE_2 = [
    "Hello! Do you have any deals or discounts right now?",
    "Add 3 sourdough breads and 2 kg of grapes.",
    "Apply any promotions that apply to my order.",
    "What delivery slots are available?",
    "I will take the evening slot. Please confirm my order.",
    "Thank you, goodbye!"
]

DIALOGUE_3 = [
    "Hi Shabo! Do you have Dragon Fruit?",
    "What about Kiwi?",
    "Okay, what is a good substitute for Kiwi from your fruits?",
    "Add 500g of strawberries and 1 pineapple then.",
    "Remove the pineapple, I changed my mind."
]

DIALOGUE_4 = [
    "Hi, what payment methods do you accept?",
    "Is there a return policy if my vegetables arrive damaged?",
    "What areas do you deliver to?",
    "How much is the delivery fee and can I get free delivery?",
    "Thanks! That is all I needed to know. Goodbye!"
]


if __name__ == "__main__":
    print("\nFreshMart Chatbot - Phase III Multi-Turn Dialogue Tests")
    print("Model: qwen2.5:1.5b via Ollama")
    print("Make sure Ollama is running before executing.\n")

    dialogues = [
        ("Dialogue 1: Browsing and Adding Items", DIALOGUE_1),
        ("Dialogue 2: Promotions and Checkout", DIALOGUE_2),
        ("Dialogue 3: Out-of-Stock and Substitution", DIALOGUE_3),
        ("Dialogue 4: Policy Inquiry", DIALOGUE_4),
    ]

    if len(sys.argv) > 1:
        idx = int(sys.argv[1]) - 1
        name, turns = dialogues[idx]
        run_dialogue(name, turns)
    else:
        for name, turns in dialogues:
            run_dialogue(name, turns)

    print_separator("All tests complete")
