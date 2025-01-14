import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

# Define a function to play sound
def play_sound():
    # Load a sound (make sure to provide a valid path to a sound file on your system)
    sound = pygame.mixer.Sound('c4.mp3')
    sound.play()

# Define the area to check (example values)
y_min = 200  # Minimum y value of the ankle area
y_max = 300  # Maximum y value of the ankle area

# Data for Left and Right Ankle Y values (example values taken from your list)
new_ankle_data = [
    (288, 292), (281, 295), (283, 296), (286, 296), (286, 279), (287, 280),
    (288, 277), (276, 279), (269, 282), (270, 283), (275, 282), (277, 273),
    (275, 265), (275, 261), (275, 244), (274, 249), (254, 250), (235, 249),
    (223, 247), (224, 245), (225, 246), (225, 238), (221, 229), (213, 231),
    (215, 225), (214, 226), (214, 225), (214, 224), (213, 223), (214, 222),
    (212, 222), (212, 222), (211, 223), (212, 223), (216, 224), (227, 224),
    (239, 225), (251, 225), (258, 227), (256, 232)
]

# Iterate through the ankle data
for left_ankle, right_ankle in ankle_data:
    # Check if both ankles' y-values are within the defined area
    if y_min <= left_ankle <= y_max and y_min <= right_ankle <= y_max:
        play_sound()
        time.sleep(1)  # Wait 1 second before checking the next values

