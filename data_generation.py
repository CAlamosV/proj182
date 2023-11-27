from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import pymunk
import math

# Define constants
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 16
SHAPE_SIDE_LENGTH = 0
FPS = 60

def simulate_motion(initial_pos, velocity, gravity, restitution, time_step):
    """
    Simulates the motion of a square object within a bounded space using Pymunk physics engine.
    The function creates a 2D physics simulation, adds a square body with specified initial properties,
    and simulates its motion for a given time step.

    Args:
        initial_pos (tuple of float): The initial position (x, y) of the square.
        velocity (tuple of float): The initial velocity (vx, vy) of the square.
        gravity (float): The gravitational acceleration applied in the simulation.
                          Positive values pull the square downward.
        restitution (float): The elasticity coefficient of the square and boundaries.
                              Values are between 0 (perfectly inelastic) and 1 (perfectly elastic).
        time_step (float): The duration for which the simulation is advanced.

    Returns:
        tuple: A tuple containing the new position (x, y) and velocity (vx, vy) of the square after the simulation step.
    """
    # Create a new space and set gravity
    space = pymunk.Space()
    space.gravity = (0, -gravity)

    # Create a body and shape for the square
    body = pymunk.Body(
        1, pymunk.moment_for_box(1, (SHAPE_SIDE_LENGTH+1, SHAPE_SIDE_LENGTH+1))
    )
    body.position = pymunk.Vec2d(*initial_pos)  # Unpack the initial_pos tuple
    body.velocity = pymunk.Vec2d(*velocity)  # Unpack the velocity tuple
    shape = pymunk.Poly.create_box(body, (SHAPE_SIDE_LENGTH+1, SHAPE_SIDE_LENGTH+1))
    shape.elasticity = restitution
    space.add(body, shape)

    # Add static lines to form boundaries of the space
    static_lines = [
        pymunk.Segment(space.static_body, (0, 0), (0, IMAGE_HEIGHT), 1),  # Left
        pymunk.Segment(
            space.static_body, (0, IMAGE_HEIGHT), (IMAGE_WIDTH, IMAGE_HEIGHT), 1
        ),  # Bottom
        pymunk.Segment(
            space.static_body, (IMAGE_WIDTH, IMAGE_HEIGHT), (IMAGE_WIDTH, 0), 1
        ),  # Right
        pymunk.Segment(space.static_body, (IMAGE_WIDTH, 0), (0, 0), 1),  # Top
    ]
    for line in static_lines:
        line.elasticity = restitution  # Set restitution for the boundaries
        space.add(line)

    # Simulate for the given time step
    space.step(time_step)

    # Return the new position and velocity
    new_pos = body.position.x, body.position.y
    new_vel = body.velocity.x, body.velocity.y

    return new_pos, new_vel


def draw_frame(position):
    """
    Draw a frame with the shape at the given position in black and white.
    """
    image = Image.new(
        "1", (IMAGE_WIDTH, IMAGE_HEIGHT), "white"
    )  # '1' for 1-bit pixels, black and white
    draw = ImageDraw.Draw(image)

    # Draw the square shape in black|
    x, y = position

    draw.rectangle([x, y, x + SHAPE_SIDE_LENGTH, y + SHAPE_SIDE_LENGTH], fill="black")

    return image


def generate_sequence(
    sequence_length,
    initial_speed,
    initial_direction,
    initial_position,
    gravity,
    coefficient_of_restitution,
    frame_rate=30,
):
    """
    Generate a sequence of images of a square object moving in a bounded space.
    """
    images = []
    position = initial_position
    velocity = (
        initial_speed * np.cos(initial_direction),
        -initial_speed * np.sin(initial_direction),
    )
    positions = []
    for frame in range(sequence_length):
        for _ in range(
            frame_rate
        ):  # Advance the simulation frame_rate times before generating an image
            position, velocity = simulate_motion(
                position,
                velocity,
                gravity,
                coefficient_of_restitution,
                1.0 / 60,  # Assuming 60 FPS for the simulation
            )

        adjusted_position = (
            position[0],
            IMAGE_HEIGHT - position[1] - SHAPE_SIDE_LENGTH,
        )
        image = draw_frame(adjusted_position)
        images.append(image)
        positions.append(position)

    return images, positions


def generate_random_sequence(
    sequence_length = 10,
    frame_rate = 30,
    speed_min = 0,
    speed_max = 10,
    direction_min = 0,
    direction_max = 2 * np.pi,
    position_x_min = 0,
    position_x_max = 16,
    position_y_min = 0,
    position_y_max = 16,
    gravity_min = 5,
    gravity_max = 10,
    restitution_min = 0.5,
    restitution_max = 1,
    ):
    """
    Generate a sequence of images of a square object moving in a bounded space with random initial properties.
    """
    # Sample each parameter
    initial_speed = np.random.uniform(speed_min, speed_max)
    initial_direction = np.random.uniform(direction_min, direction_max)
    initial_position_x = np.random.uniform(position_x_min, position_x_max)
    initial_position_y = np.random.uniform(position_y_min, position_y_max)
    gravity = np.random.uniform(gravity_min, gravity_max)
    coefficient_of_restitution = np.random.uniform(restitution_min, restitution_max)

    # Generate the sequence
    sequence, positions = generate_sequence(
        sequence_length,
        initial_speed,
        initial_direction,
        (initial_position_x, initial_position_y),
        gravity,
        coefficient_of_restitution,
        frame_rate,
    )

    return sequence, positions

def display_sequence(sequence):
    # Display the images side by side with boundaries between frames
    fig, axes = plt.subplots(
        1, len(sequence), figsize=(20, 2)
    )  # Adjust figsize as needed

    # Adding a small space between images for clear separation
    plt.subplots_adjust(wspace=0.1)  # Adjust space as needed

    for ax, img in zip(axes, sequence):
        ax.imshow(img)
        ax.axis("on")  # Turn on axis to create a boundary
        ax.set_xticks([])
        ax.set_yticks([])  # Remove tick marks

    plt.show()
