import cv2

def get_smallest_widt_height(img_list: list) -> (int, int):
    min_width = 1000000
    min_height = 1000000
    for img_path in img_list:
        img = Image.open(img_path)
        width, height = img.size
        if width < min_width:
            min_width = width
        if height < min_height:
            min_height = height
    return min_width, min_height