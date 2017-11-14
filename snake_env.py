import numpy as np
import random, pygame, sys
from pygame.locals import *
FPS = 5
##WINDOWWIDTH = 640
#WINDOWHEIGHT = 480
WINDOWWIDTH = 640
WINDOWHEIGHT = 480
CELLSIZE = 40
assert WINDOWWIDTH % CELLSIZE == 0, "Window width must be a multiple of cell size."
assert WINDOWHEIGHT % CELLSIZE == 0, "Window height must be a multiple of cell size."
CELLWIDTH = int(WINDOWWIDTH / CELLSIZE)
CELLHEIGHT = int(WINDOWHEIGHT / CELLSIZE)

#             R    G    B
WHITE     = (255, 255, 255)
BLACK     = (  0,   0,   0)
RED       = (255,   0,   0)
GREEN     = (  0, 255,   0)
DARKGREEN = (  0, 155,   0)
DARKGRAY  = ( 40,  40,  40)
BGCOLOR = BLACK
UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'
ACTIONS = [UP, DOWN, LEFT, RIGHT]
HEAD = 0 # syntactic sugar: index of the worm's head
#global FPSCLOCK, DISPLAYSURF, BASICFONT
pygame.init()
FPSCLOCK = pygame.time.Clock()
DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
pygame.display.set_caption('Snaky')

def examine_direction(temp , direction):
    if direction == UP:
        if temp == DOWN:
            return False
    elif direction == RIGHT:
        if temp == LEFT:
            return False
    elif direction == LEFT:
        if temp == RIGHT:
            return False
    elif direction == DOWN:
        if temp == UP:
            return False
    return True

def drawPressKeyMsg():
    pressKeySurf = BASICFONT.render('Press a key to play.', True, DARKGRAY)
    pressKeyRect = pressKeySurf.get_rect()
    pressKeyRect.topleft = (WINDOWWIDTH - 200, WINDOWHEIGHT - 30)
    DISPLAYSURF.blit(pressKeySurf, pressKeyRect)

def checkForKeyPress():
    if len(pygame.event.get(QUIT)) > 0:
        terminate()
    keyUpEvents = pygame.event.get(KEYUP)
    if len(keyUpEvents) == 0:
        return None
    if keyUpEvents[0].key == K_ESCAPE:
        terminate()
    return keyUpEvents[0].key

def terminate():
    pygame.quit()
    sys.exit()

def getRandomLocation(worm):
    temp = {'x': random.randint(0, CELLWIDTH - 1), 'y': random.randint(0, CELLHEIGHT - 1)}
    while tt_not_ok(temp, worm):
        temp = {'x': random.randint(0, CELLWIDTH - 1), 'y': random.randint(0, CELLHEIGHT - 1)}
    return temp

def tt_not_ok(temp, worm):
    for body in worm:
        if temp['x'] == body['x'] and temp['y'] == body['y']:
            return True
    return False

def showGameOverScreen():
    gameOverFont = pygame.font.Font('freesansbold.ttf', 150)
    gameSurf = gameOverFont.render('Game', True, WHITE)
    overSurf = gameOverFont.render('Over', True, WHITE)
    gameRect = gameSurf.get_rect()
    overRect = overSurf.get_rect()
    gameRect.midtop = (WINDOWWIDTH / 2, 10)
    overRect.midtop = (WINDOWWIDTH / 2, gameRect.height + 10 + 25)
    DISPLAYSURF.blit(gameSurf, gameRect)
    DISPLAYSURF.blit(overSurf, overRect)
    drawPressKeyMsg()
    pygame.display.update()
    pygame.time.wait(500)
    #checkForKeyPress() # clear out any key presses in the event queue
    while True:
        if checkForKeyPress():
            pygame.event.get() # clear event queue
            return

def drawScore(score):
    scoreSurf = BASICFONT.render('Score: %s' % (score), True, WHITE)
    scoreRect = scoreSurf.get_rect()
    scoreRect.topleft = (WINDOWWIDTH - 120, 10)
    DISPLAYSURF.blit(scoreSurf, scoreRect)

def drawWorm(wormCoords):
    for coord in wormCoords:
        x = coord['x'] * CELLSIZE
        y = coord['y'] * CELLSIZE
        wormSegmentRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
        pygame.draw.rect(DISPLAYSURF, DARKGREEN, wormSegmentRect)
        wormInnerSegmentRect = pygame.Rect(x + 4, y + 4, CELLSIZE - 8, CELLSIZE - 8)
        pygame.draw.rect(DISPLAYSURF, GREEN, wormInnerSegmentRect)
def drawApple(coord):
    x = coord['x'] * CELLSIZE
    y = coord['y'] * CELLSIZE
    appleRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
    pygame.draw.rect(DISPLAYSURF, RED, appleRect)

def drawGrid():
    for x in range(0, WINDOWWIDTH, CELLSIZE): # draw vertical lines
        pygame.draw.line(DISPLAYSURF, DARKGRAY, (x, 0), (x, WINDOWHEIGHT))
    for y in range(0, WINDOWHEIGHT, CELLSIZE): # draw horizontal lines
        pygame.draw.line(DISPLAYSURF, DARKGRAY, (0, y), (WINDOWWIDTH, y))

class GameEnv(object):
    def reset(self):
        self.startx = random.randint(5, CELLWIDTH - 6)
        self.starty = random.randint(5, CELLHEIGHT - 6)
        self.direction = RIGHT
        self.wormCoords = [{'x': self.startx, 'y': self.starty},
                           {'x': self.startx - 1, 'y': self.starty},
                           {'x': self.startx - 2, 'y': self.starty}]
        self.direction = RIGHT
        self.apple = getRandomLocation(self.wormCoords)

    def step(self, action):
        reward = 0.1
        terminal = False
        action = ACTIONS[np.argmax(action)]
        self.pre_direction = self.direction
        if (action == "left") and self.direction != RIGHT:
            self.direction = LEFT
        elif (action == "right") and self.direction != LEFT:
            self.direction = RIGHT
        elif (action == "up") and self.direction != DOWN:
            self.direction = UP
        elif (action == "down") and self.direction != UP:
            self.direction = DOWN
        if self.wormCoords[HEAD]['x'] == -1 or self.wormCoords[HEAD]['x'] == CELLWIDTH or self.wormCoords[HEAD][
            'y'] == -1 or self.wormCoords[HEAD]['y'] == CELLHEIGHT:
            reward = -1
            terminal = True
            self.reset()
            image_data = self.render()
            return image_data, reward, terminal

        for wormBody in self.wormCoords[1:]:
            if wormBody['x'] == self.wormCoords[HEAD]['x'] and wormBody['y'] == self.wormCoords[HEAD]['y']:
                reward = -1
                terminal = True
                self.reset()
                image_data = self.render()
                return image_data, reward, terminal
        # check if worm has eaten an apply
        if self.wormCoords[HEAD]['x'] == self.apple['x'] and self.wormCoords[HEAD]['y'] == self.apple['y']:
            # don't remove worm's tail segment
            reward = 1
            self.apple = getRandomLocation(self.wormCoords)  # set a new apple somewhere
        else:
            del self.wormCoords[-1]  # remove worm's tail segment
        # move the worm by adding a segment in the direction it is moving
        if not examine_direction(self.direction, self.pre_direction):
            self.direction = self.pre_direction
        if self.direction == UP:
            newHead = {'x': self.wormCoords[HEAD]['x'], 'y': self.wormCoords[HEAD]['y'] - 1}
        elif self.direction == DOWN:
            newHead = {'x': self.wormCoords[HEAD]['x'], 'y': self.wormCoords[HEAD]['y'] + 1}
        elif self.direction == LEFT:
            newHead = {'x': self.wormCoords[HEAD]['x'] - 1, 'y': self.wormCoords[HEAD]['y']}
        elif self.direction == RIGHT:
            newHead = {'x': self.wormCoords[HEAD]['x'] + 1, 'y': self.wormCoords[HEAD]['y']}
        self.wormCoords.insert(0, newHead)
        image_data = self.render()
        return image_data, reward, terminal

    def render(self):
        DISPLAYSURF.fill(BGCOLOR)
        drawGrid()
        drawWorm(self.wormCoords)
        drawApple(self.apple)
        drawScore(len(self.wormCoords) - 3)
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        #print(image_data[100][1],image_data.shape, "------")
        #image_data = ""
        return image_data

def main_ai():
    obj = GameEnv()
    actions = ["down", "up", "right", "left"]
    obj.reset()
    for i in range(10):
        done = False
        while True:
            #obj.render()
            do_nothing = np.zeros(4)
            index = random.choice(range(4))
            do_nothing[index] = 1
            # observation_, reward, done = obj.step(action)
            states, reward, done = obj.step(do_nothing)
            if done:
                break
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main_ai()