"""
Evaluation dataset for Bloomington Tourist RAG system.

Each entry has:
- question: what a tourist might ask
- expected_topic: which file should be retrieved
- ground_truth: the correct answer (for RAGAS context_recall)
- keywords: words that must appear in a good answer
"""

EVAL_QUESTIONS = [
    {
        "id": "hotel_001",
        "question": "What budget hotels are available near Indiana University?",
        "expected_topic": "hotels_and_accommodation",
        "ground_truth": "Several budget hotels are available near Indiana University including Motel 6, Super 8 by Wyndham, and Comfort Inn. Prices typically range from $60-$100 per night depending on season.",
        "keywords": ["hotel", "budget", "university", "indiana"]
    },
    {
        "id": "hotel_002",
        "question": "Which hotels in Bloomington have a pool and free breakfast?",
        "expected_topic": "hotels_and_accommodation",
        "ground_truth": "Hotels like Holiday Inn Express and Hampton Inn in Bloomington offer both a pool and complimentary breakfast. Prices range from $100-$150 per night.",
        "keywords": ["pool", "breakfast", "hotel"]
    },
    {
        "id": "hotel_003",
        "question": "What is the average cost of a 3-star hotel in Bloomington?",
        "expected_topic": "hotels_and_accommodation",
        "ground_truth": "On average, a 3-star hotel in Bloomington costs around $210 per night.",
        "keywords": ["3-star", "cost", "average", "210"]
    },
    {
        "id": "hotel_004",
        "question": "Are there pet-friendly hotels in Bloomington Indiana?",
        "expected_topic": "hotels_and_accommodation",
        "ground_truth": "Many hotels in Bloomington allow pets. La Quinta Inn and Suites sometimes has pet-friendly options. Always check the hotel's individual pet policy for fees and size restrictions.",
        "keywords": ["pet", "hotel", "bloomington"]
    },
    {
        "id": "hotel_005",
        "question": "What extended stay hotel options exist in Bloomington?",
        "expected_topic": "hotels_and_accommodation",
        "ground_truth": "Candlewood Suites Bloomington is an IHG hotel that offers extended-stay accommodations with kitchenettes.",
        "keywords": ["extended", "stay", "candlewood", "kitchenette"]
    },
    {
        "id": "hotel_006",
        "question": "What are the cheapest hotels in Bloomington Indiana?",
        "expected_topic": "hotels_and_accommodation",
        "ground_truth": "The cheapest hotel options in Bloomington start from $61 per night. Super 8 by Wyndham Bloomington is a potentially budget-friendly option.",
        "keywords": ["cheapest", "hotel", "bloomington", "budget"]
    },
    {
        "id": "hotel_007",
        "question": "Which IHG hotels are near Indiana University campus?",
        "expected_topic": "hotels_and_accommodation",
        "ground_truth": "The Holiday Inn Bloomington-University Area is the closest IHG hotel to Indiana University. It typically offers a pool, fitness center, and free breakfast.",
        "keywords": ["IHG", "holiday inn", "indiana university"]
    },
    {
        "id": "restaurant_001",
        "question": "Where can I find good restaurants in downtown Bloomington?",
        "expected_topic": "restaurants_and_dining",
        "ground_truth": "Downtown Bloomington has a vibrant food scene with a mix of restaurants, cafes, and bars. The area is known for its eclectic dining options.",
        "keywords": ["restaurant", "downtown", "bloomington", "dining"]
    },
    {
        "id": "restaurant_002",
        "question": "What dining options are available near Indiana University?",
        "expected_topic": "restaurants_and_dining",
        "ground_truth": "There are many dining options near Indiana University in Bloomington including local restaurants and cafes along Kirkwood Avenue.",
        "keywords": ["dining", "restaurant", "indiana university"]
    },
    {
        "id": "restaurant_003",
        "question": "Are there any farm to table restaurants in Bloomington?",
        "expected_topic": "restaurants_and_dining",
        "ground_truth": "FARMbloomington offers a farm-to-table dining experience with seasonal menus and a cozy atmosphere.",
        "keywords": ["farm", "restaurant", "bloomington", "seasonal"]
    },
    {
        "id": "event_001",
        "question": "What events happen at Indiana University Bloomington?",
        "expected_topic": "events_and_university",
        "ground_truth": "Indiana University hosts sports games at Assembly Hall and Memorial Stadium, performances by the IU Auditorium and Jacobs School of Music, and various cultural festivals.",
        "keywords": ["indiana university", "events", "assembly hall", "auditorium"]
    },
    {
        "id": "event_002",
        "question": "How can I find out about student housing near IU campus?",
        "expected_topic": "events_and_university",
        "ground_truth": "You can find information about student housing near IU campus through the Indiana University housing office or by checking local rental listings.",
        "keywords": ["student", "housing", "campus", "indiana university"]
    },
    {
        "id": "event_003",
        "question": "What is the process for renting event space at Indiana University?",
        "expected_topic": "events_and_university",
        "ground_truth": "You can contact Indiana University directly to inquire about renting event spaces on campus.",
        "keywords": ["renting", "event", "indiana university", "space"]
    },
    {
        "id": "hotel_008",
        "question": "How can I find last minute hotel deals in Bloomington?",
        "expected_topic": "hotels_and_accommodation",
        "ground_truth": "For last-minute hotel deals in Bloomington use online travel agencies like Expedia, Booking.com, Kayak, and Hotels.com. You can also call hotels directly for unsold rooms at reduced rates.",
        "keywords": ["last minute", "hotel", "deals", "bloomington"]
    },
    {
        "id": "hotel_009",
        "question": "Are there cabin or campground rentals near Bloomington Indiana?",
        "expected_topic": "hotels_and_accommodation",
        "ground_truth": "For cabin or campground rentals outside downtown Bloomington search on Airbnb, VRBO, and Hipcamp. Prices range from $75-$200 per night for cabins or $20-$50 for campsites.",
        "keywords": ["cabin", "campground", "bloomington", "rental"]
    }
]


def get_questions():
    return [item["question"] for item in EVAL_QUESTIONS]


def get_expected_topics():
    return {
        item["question"]: item["expected_topic"]
        for item in EVAL_QUESTIONS
    }


def get_ground_truths():
    return {
        item["question"]: item["ground_truth"]
        for item in EVAL_QUESTIONS
    }