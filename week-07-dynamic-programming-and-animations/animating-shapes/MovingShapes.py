from Shapes import *
from pylab import random as r
import math

####################################################
# Superclass for all moving shapes
####################################################

class MovingShape:
    def __init__(self, frame, shape, diameter):
        """
        frame   : Frame object that knows the window size
        shape   : string ('square', 'circle', 'diamond')
        diameter: visual size of the shape
        """
        self.frame = frame
        self.shape = shape
        self.diameter = diameter

        # Underlying drawable Shape (from Shapes.py)
        self.figure = Shape(shape, diameter)

        # -------------------------------------------------
        # random start position INSIDE the frame
        # -------------------------------------------------
        # First compute min/max coordinates that are safe
        # for this shape type (no overlap with border).
        self.set_min_max()

        # Choose a random (x, y) inside those bounds
        self.x = self.minx + r() * (self.maxx - self.minx)
        self.y = self.miny + r() * (self.maxy - self.miny)

        # Move the turtle to this starting position
        self.goto_curr_xy()

        # -------------------------------------------------
        # random velocity components
        # -------------------------------------------------
        # Base speed between 5 and 15
        speed_x = 5 + 10 * r()
        speed_y = 5 + 10 * r()

        # Randomly flip sign (50% chance) so some go left/up
        if r() < 0.5:
            speed_x = -speed_x
        if r() < 0.5:
            speed_y = -speed_y

        self.dx = speed_x
        self.dy = speed_y

    # -----------------------------------------------------

    def set_min_max(self):
        """
        Compute min/max x and y values this shape is allowed
        to occupy. For squares and circles we base this on
        the given diameter. Diamonds override this method.
        """
        d = self.diameter
        # Keep the whole shape inside the border
        self.minx = d / 2
        self.maxx = self.frame.width - d / 2
        self.miny = d / 2
        self.maxy = self.frame.height - d / 2

    # -----------------------------------------------------

    def goto_curr_xy(self):
        """Move the underlying turtle to the current (x, y)."""
        self.figure.goto(self.x, self.y)

    # -----------------------------------------------------

    def moveTick(self):
        """
        One 'tick' of the animation clock.
        Update position by (dx, dy), then check for bounces
        against the frame walls. If a bounce happens, reverse
        the appropriate velocity component and report it.
        """
        # Update position
        self.x += self.dx
        self.y += self.dy

        bounced = False

        # bounce off vertical walls (left/right)
        if self.x < self.minx:
            self.x = self.minx
            self.dx = -self.dx
            bounced = True
        elif self.x > self.maxx:
            self.x = self.maxx
            self.dx = -self.dx
            bounced = True

        # bounce off horizontal walls (top/bottom)
        if self.y < self.miny:
            self.y = self.miny
            self.dy = -self.dy
            bounced = True
        elif self.y > self.maxy:
            self.y = self.maxy
            self.dy = -self.dy
            bounced = True

        # If we bounced, print the chatty message
        if bounced:
            self.report_bounce()

        # Finally move the shape in the graphics window
        self.goto_curr_xy()

    # -----------------------------------------------------

    def report_bounce(self):
        """
        'chatty' shapes.
        Called each time the shape bounces.  Uses my_area()
        which is defined separately in each subclass.
        """
        area = self.my_area()
        print(f"I'm a bouncing {self.shape} - my area is {area:.0f} sq. units")

    # -----------------------------------------------------

    def my_area(self):
        """
        Placeholder; subclasses each provide their own
        implementation to compute area.  (Square, Circle,
        Diamond override this.)
        """
        return 0

    # -----------------------------------------------------

    def check_collide(self, other):
        """
        default interaction behaviour between shapes.
        The base class does nothing – only some subclasses
        (e.g. Square) override this.
        """
        pass


####################################################
# Subclasses
####################################################

class Square(MovingShape):
    def __init__(self, frame, diameter):
        super().__init__(frame, 'square', diameter)

    def my_area(self):
        # Area of a square with side length = diameter
        return self.diameter ** 2

    # -------------------------------------------------

    def check_collide(self, other):
        """
        Colliding squares: elastic collision.
        We only care if the other shape is also a Square.
        The idea:
            * If their x and y distances are both <= the
              sum of their "radii" (diameter/2), we say
              there is a collision.
            * If the horizontal gap is larger than vertical,
              treat as top/bottom -> swap dy components.
            * Otherwise treat as side/side -> swap dx.
        """
        if not isinstance(other, Square):
            return

        # Distance between centres
        dx = abs(self.x - other.x)
        dy = abs(self.y - other.y)

        # Collision threshold (very simple approximation)
        thresh = (self.diameter + other.diameter) / 2.0

        if dx <= thresh and dy <= thresh:
            # Determine if it's more horizontal or vertical
            if dx > dy:
                # Top/bottom type collision -> swap vertical speeds
                self.dy, other.dy = other.dy, self.dy
            else:
                # Side/side collision -> swap horizontal speeds
                self.dx, other.dx = other.dx, self.dx


####################################################

class Diamond(MovingShape):
    def __init__(self, frame, diameter):
        super().__init__(frame, 'diamond', diameter)

    # -------------------------------------------------

    def set_min_max(self):
        """
        Diamonds vs Squares.
        A diamond is a square tilted by 45 degrees, so its
        'practical' diameter (corner-to-corner) is larger:
            d2 = sqrt(2) * d1
        We use this larger value when computing min/max so
        that corners do not cross the border.
        """
        d2 = (2 ** 0.5) * self.diameter
        self.minx = d2 / 2
        self.maxx = self.frame.width - d2 / 2
        self.miny = d2 / 2
        self.maxy = self.frame.height - d2 / 2

    # -------------------------------------------------

    def my_area(self):
        # Diamond has same area as the unrotated square
        return self.diameter ** 2


####################################################

class Circle(MovingShape):
    def __init__(self, frame, diameter):
        super().__init__(frame, 'circle', diameter)

    def my_area(self):
        # Area of a circle with diameter d:  π (d/2)^2
        radius = self.diameter / 2.0
        return math.pi * radius * radius

####################################################
