"""Mobility classes for HSSD object categories (used by the rearrangement planner and the figures).

structural : never changes between visits (architecture, fixtures, built-in appliances)
heavy      : large furniture; moved only in strong rearrangements (refurnishing)
light      : furniture that is moved routinely (chairs, plants, lamps, bins, boxes)
clutter    : small objects that move at almost every visit (dishes, books, cushions, toiletries)
decor      : wall-mounted decoration; removed or replaced rather than translated
"""
STRUCTURAL = {
    "window", "door", "opening", "railing", "beam", "fence", "stairs", "staircase", "counter", "countertop", "kitchen_counter",
    "carpet", "mat", "rug", "curtain", "blind", "ceiling_lamp", "wall_lamp", "chandelier", "pendant_lamp", "socket", "switch",
    "toilet", "sink", "bathtub", "shower", "showerhead", "faucet", "shower_faucet", "toilet_paper_holder", "toilet_flush_plate",
    "towel_rack", "range_hood", "air_conditioner", "fridge", "dishwasher", "washer_dryer", "clothes_dryer", "oven", "stove",
    "cooktop", "microwave", "swimming_pool", "fireplace", "mantel", "radiator", "vent", "smoke_detector", "thermostat", "wall_hook_rack",
    "greenhouse", "shed", "playhouse", "play_area", "trampoline", "net", "dining_area", "unknown", "balcony", "column", "pillar",
    "wardrobe_builtin", "bar", "kitchen", "bath", "wall_sticker", "screen", "projector", "mirror", "wall_sign",
}
HEAVY = {
    "bed", "couch", "sofa", "sectional", "wardrobe", "shelves", "shelf", "bookshelf", "cabinet", "chest_of_drawers", "dresser",
    "table", "dining_table", "desk", "tv", "tv_stand", "sideboard", "buffet", "piano", "treadmill", "gym_equipment", "exercise_bike",
    "car", "motorcycle", "stand", "drawer_unit", "counter_stool", "workbench", "bench", "loudspeaker", "subwoofer", "media_player",
    "armoire", "crib", "bunk_bed", "daybed", "chaise", "kitchen_island",
}
LIGHT = {
    "chair", "armchair", "stool", "ottoman", "potted_plant", "plant", "floor_lamp", "table_lamp", "lamp", "trashcan", "hamper",
    "storage_box", "box", "basket", "shoe_rack", "drying_rack", "clothing_rack", "ladder", "vacuum", "serving_cart", "cart",
    "ironing_board", "bicycle", "side_table", "coffee_table", "nightstand", "end_table", "bar_stool", "rocking_chair", "bean_bag",
    "suitcase", "luggage", "stroller", "umbrella", "barbecue", "firepit", "guitar", "microphone", "printer", "fan", "heater",
    "bathroom_scale", "step_stool", "toy", "ball", "soccer_ball", "wagon", "pet_bed", "cooler",
}
CLUTTER = {
    "drinkware", "cup", "mug", "glass", "bottle", "plate", "bowl", "dish", "pot", "pan", "kettle", "teapot", "tray", "book", "books",
    "magazine", "vase", "cushion", "pillow", "blanket", "towel", "board_game", "laptop", "tablet", "phone", "keyboard", "mouse", "monitor",
    "picture_frame", "mantel_clock", "clock", "toiletry", "soap_dish", "soap", "toilet_brush", "tissue_box", "candle", "sculpture",
    "figurine", "globe", "flower", "bouquet", "fruit", "food", "letter", "paper", "notebook", "pen", "remote", "controller", "headphones",
    "camera", "shoe", "shoes", "bag", "backpack", "hat", "clothes", "timer", "jar", "can", "container", "cutting_board", "utensil",
    "knife", "spoon", "fork", "toaster", "blender", "coffee_maker", "lunch_box", "trophy", "toy_car", "doll", "plush",
}
DECOR = {"picture", "painting", "poster", "wall_decor", "wall_clock", "wall_art", "tapestry", "calendar", "whiteboard", "bulletin_board"}

# sampling weight of each class (relative probability of being changed at a given level)
CLASS_WEIGHT = {"clutter": 1.0, "light": 0.8, "decor": 0.5, "heavy": 0.3, "structural": 0.0}
# operation mix per class: relocate (new free spot in the same room), jitter (small push), remove, swap (with same-class object)
CLASS_OPS = {
    "clutter": {"relocate": 0.5, "jitter": 0.2, "remove": 0.25, "swap": 0.05},
    "light": {"relocate": 0.5, "jitter": 0.25, "remove": 0.15, "swap": 0.10},
    "decor": {"relocate": 0.0, "jitter": 0.0, "remove": 0.6, "swap": 0.4},
    "heavy": {"relocate": 0.4, "jitter": 0.35, "remove": 0.15, "swap": 0.10},
}
CLASS_COLOR = {"structural": "#9e9e9e", "heavy": "#5b7bd5", "light": "#e6a23c", "clutter": "#d9534f", "decor": "#8e5bd5"}


MOUNTED_FIXED = {"cabinet", "shelves", "shelf", "wall_shelves", "wall_unit", "bathroom_vanity_units", "heating_system", "string_lights",
                 "tv", "counter", "wardrobe", "drawer_unit", "kids'_storage", "rack"}


def mobility_class(category: str, maxdim: float = -1.0, super_category: str = "", z_bottom: float = 0.0) -> str:
    c = (category or "unknown").lower()
    if c in STRUCTURAL:
        return "structural"
    if c in ("wall_shelves", "wall_unit", "bathroom_vanity_units", "heating_system", "string_lights", "gazebos", "garage_and_storage",
             "dining_area", "play_area", "swimming_pool", "fence", "fireplace"):
        return "structural"
    if c in MOUNTED_FIXED and z_bottom > 0.3:          # wall-mounted cabinets / shelves / TVs are fixed
        return "structural"
    if c in DECOR:
        return "decor"
    if c in CLUTTER:
        return "clutter"
    if c in LIGHT:
        return "light"
    if c in HEAVY:
        return "heavy"
    s = (super_category or "").lower()
    if s in ("dining_ware", "kitchenware", "food", "toiletries", "stationery", "electronics_small"):
        return "clutter"
    if s in ("seating_furniture",):
        return "light" if 0 < maxdim < 1.2 else "heavy"
    if s in ("storage_furniture", "support_furniture", "sleeping_furniture", "vehicle", "large_appliance"):
        return "heavy"
    if s in ("decor",):
        return "clutter" if 0 < maxdim < 0.8 else "decor"
    if s in ("arch", "bathroom_fixtures", "lighting", "floor_covering", "curtain", "window", "door"):
        return "structural"
    if s in ("plant",):
        return "light"
    # unknown category: decide by size
    if maxdim < 0:
        return "structural"
    if maxdim < 0.5:
        return "clutter"
    if maxdim < 1.3:
        return "light"
    return "heavy"
