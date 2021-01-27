import pygame
import math
import json
import sys

WIDTH = 400
HEIGHT = 400


def draw_circles(number, radius, distance):


    root = math.sqrt(number)
    if int(root + 0.5) ** 2 == number:
        rows = math.sqrt(number)
    else:
        rows = math.ceil(math.sqrt(number))

    pixels_per_unit = math.floor(WIDTH / (rows * radius * 2 + rows * distance))

    actual_size = (int((rows * radius * 2 + rows * distance) * pixels_per_unit))

    screen = pygame.display.set_mode((actual_size, actual_size + 100))

    pygame.draw.rect(screen, data["bg color"], (0, 0, actual_size, actual_size + 5))

    for x in range(int(rows)):
        for y in range(int(rows)):

            x_pos = ((0.5 * distance + radius) * pixels_per_unit + y * ((distance + radius * 2) * pixels_per_unit))
            y_pos = ((0.5 * distance + radius) * pixels_per_unit + x * ((distance + radius * 2) * pixels_per_unit))
            r = (pixels_per_unit * radius)
            if data["kółka pełne - false / true"]:
                pygame.draw.circle(screen, data["circle color"], (int(x_pos), int(y_pos)), int(r))
            else:
                pygame.draw.circle(screen, data["circle color"], (int(x_pos), int(y_pos)), int(r), 2)
            if x * rows + y == number - 1:
                break


    pygame.draw.rect(screen, (255, 255, 255), (0, actual_size,
                                               actual_size, 100))

    rectangle_size = int((rows * radius * 2 + (rows - 1) * distance) * pixels_per_unit) ** 2
    circle_coverage = number * math.pi * (pixels_per_unit * radius) ** 2

    print(rectangle_size)
    print(circle_coverage)
    coverage = circle_coverage / rectangle_size * 100
    textsurface = myfont.render("Circle covering {0:.2f}% of the rectangle".format(coverage), False, (0, 0, 0))
    screen.blit(textsurface, (10, actual_size + 25))
    pygame.display.flip()



pygame.init()
pygame.font.init()
# print(pygame.font.get_fonts())
myfont = pygame.font.SysFont('carlito', 24)

dict = {
    "number of circles": 9,
    "radius": 1,
    "distance": 0.2,
    "circle color": (102, 91, 117),
    "bg color": (160, 148, 150),
    "kółka pełne - false / true": False

}

try:
    with open('config.json') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Byku stwórz se config")
    print("Przykladowy ci sie wygenerował")
    input("Kliknij Enter no nie")
    with open('config.json', 'w') as fp:
        json.dump(dict, fp)
    sys.exit()


draw_circles(data["number of circles"], data["radius"], data["distance"])

running = True


while running:
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type is pygame.QUIT:
            running = False