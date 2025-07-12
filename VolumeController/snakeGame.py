import cv2
import dlib
import pygame
import random
import imutils
import time

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 640, 480
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Face Controlled Snake")

# Snake settings
snake_pos = [100, 50]
snake_body = [[100, 50]]
snake_direction = 'RIGHT'
change_to = snake_direction
speed = 15

# Food settings
food_pos = [random.randrange(1, WIDTH//10) * 10,
            random.randrange(1, HEIGHT//10) * 10]
food_spawn = True

# Game settings
clock = pygame.time.Clock()
score = 0

def game_over():
    font = pygame.font.SysFont('times new roman', 50)
    text = font.render(f'Game Over! Score: {score}', True, (255, 0, 0))
    win.blit(text, [WIDTH//6, HEIGHT//3])
    pygame.display.flip()
    time.sleep(2)
    pygame.quit()
    quit()

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    direction = snake_direction

    for face in faces:
        landmarks = predictor(gray, face)

        nose = landmarks.part(30)
        x = nose.x
        y = nose.y

        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        if abs(x - frame_center_x) > 25:
            if x < frame_center_x:
                direction = 'LEFT'
            else:
                direction = 'RIGHT'
        elif abs(y - frame_center_y) > 25:
            if y < frame_center_y:
                direction = 'UP'
            else:
                direction = 'DOWN'

    # Update snake direction
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

    # Snake body growing mechanism
    snake_body.insert(0, list(snake_pos))
    if snake_pos == food_pos:
        score += 1
        food_spawn = False
    else:
        snake_body.pop()

    if not food_spawn:
        food_pos = [random.randrange(1, WIDTH//10) * 10,
                    random.randrange(1, HEIGHT//10) * 10]
    food_spawn = True

    # Fill background
    win.fill((0, 0, 0))

    # Draw snake
    for pos in snake_body:
        pygame.draw.rect(win, (0, 255, 0), pygame.Rect(
            pos[0], pos[1], 10, 10))

    # Draw food
    pygame.draw.rect(win, (255, 0, 0), pygame.Rect(
        food_pos[0], food_pos[1], 10, 10))

    # Game Over conditions
    if snake_pos[0] < 0 or snake_pos[0] > WIDTH-10:
        game_over()
    if snake_pos[1] < 0 or snake_pos[1] > HEIGHT-10:
        game_over()

    for block in snake_body[1:]:
        if snake_pos == block:
            game_over()

    pygame.display.update()
    clock.tick(speed)

    # Exit on pygame QUIT
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            quit()
