def transformYokeDistToAngle(yoke_dist):
    dist = abs(12.1 - yoke_dist)

    angle = dist / 2 * -15

    return angle / 360 * 2 * 3.1415

