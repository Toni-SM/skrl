import warp as wp


def ones_(array: wp.array) -> wp.array:
    array.fill_(1.0)
    return array
