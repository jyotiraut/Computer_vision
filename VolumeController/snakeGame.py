import cv2
import mediapipe as mp
import pygame
import math
import time
import sys

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 640, 480
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Face-Controlled Snake")
clock = pygame.time.Clock()

# Snake variables
snake_pos = [100, 50]
snake_body = [[100, 50]]
snake_direction = 'RIGHT'
speed = 15
game_over = False

# Food variables
import random
food_pos = [random.randrange(1, WIDTH//10)*10, random.randrange(1, HEIGHT//10)*10]
food_spawn = True

def draw_snake():
    win.fill((0, 0, 0))
    for pos in snake_body:
        pygame.draw.rect(win, (0, 255, 0), pygame.Rect(pos[0], pos[1], 10, 10))
    pygame.draw.rect(win, (255, 0, 0), pygame.Rect(food_pos[0], food_pos[1], 10, 10))
    pygame.display.update()

def end_game():
    font = pygame.font.SysFont('times new roman', 40)
    text = font.render('Game Over!', True, (255, 0, 0))
    win.blit(text, [WIDTH//3, HEIGHT//3])
    pygame.display.flip()
    time.sleep(2)
    pygame.quit()
    sys.exit()

while True:
    # Read webcam frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get face landmarks
    results = face_mesh.process(rgb)
    direction = snake_direction

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose = face_landmarks.landmark[1]  # nose tip
            h, w, _ = frame.shape
            x = int(nose.x * w)
            y = int(nose.y * h)

            cx, cy = w // 2, h // 2
            dx, dy = x - cx, y - cy

            # Determine direction based on nose offset
            if abs(dx) > abs(dy):
                direction = 'LEFT' if dx < -20 else 'RIGHT' if dx > 20 else direction
            else:
                direction = 'UP' if dy < -20 else 'DOWN' if dy > 20 else direction
            break

    # Set new direction if valid
    if direction == 'UP' and not snake_direction == 'DOWN':
        snake_direction = 'UP'
    if direction == 'DOWN' and not snake_direction == 'UP':
        snake_direction = 'DOWN'
    if direction == 'LEFT' and not snake_direction == 'RIGHT':
        snake_direction = 'LEFT'
    if direction == 'RIGHT' and not snake_direction == 'LEFT':
        snake_direction = 'RIGHT'

    # Move snake
    if snake_direction == 'UP':
        snake_pos[1] -= 10
    if snake_direction == 'DOWN':
        snake_pos[1] += 10
    if snake_direction == 'LEFT':
        snake_pos[0] -= 10
    if snake_direction == 'RIGHT':
        snake_pos[0] += 10

    snake_body.insert(0, list(snake_pos))
    if snake_pos == food_pos:
        food_spawn = False
    else:
        snake_body.pop()

    if not food_spawn:
        food_pos = [random.randrange(1, WIDTH//10)*10, random.randrange(1, HEIGHT//10)*10]
    food_spawn = True

    # Check collisions
    if snake_pos[0] < 0 or snake_pos[0] > WIDTH-10 or \
       snake_pos[1] < 0 or snake_pos[1] > HEIGHT-10 or \
       snake_pos in snake_body[1:]:
        end_game()

    draw_snake()
    clock.tick(speed)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()
