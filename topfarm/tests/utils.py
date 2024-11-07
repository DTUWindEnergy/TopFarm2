import numpy as np


def __assert_equal_unordered(array1, array2, rtol=1e-5, atol=1e-8):
    """
    Assert if two (N, 2) numpy arrays contain the same pairs of values,
    irrespective of order along the 0th dimension, with floating point tolerance.

    Parameters:
    array1 (np.ndarray): First array to compare, with shape (N, 2).
    array2 (np.ndarray): Second array to compare, with shape (N, 2).
    rtol (float): The relative tolerance for floating-point comparison.
    atol (float): The absolute tolerance for floating-point comparison.

    Raises:
    AssertionError: If arrays are not equal irrespective of order along the 0th dimension.
    """
    # Check shapes
    if array1.shape != array2.shape:
        raise AssertionError(f"Shapes do not match: {array1.shape} != {array2.shape}")

    # Sort both arrays along the 0th dimension
    sorted_array1 = np.sort(array1, axis=0)
    sorted_array2 = np.sort(array2, axis=0)

    # Check if the sorted arrays are close within the specified tolerance
    if not np.allclose(sorted_array1, sorted_array2, rtol=rtol, atol=atol):
        raise AssertionError("Arrays are not equal, even when unordered.")


if __name__ == "__main__":
    # make sure the function working as intended
    def test___assert_equal_unordered():
        for _ in range(100):
            array1 = np.random.rand(100, 2)
            array2 = np.random.rand(100, 2)
            try:
                __assert_equal_unordered(array1, array2)
                assert False, "Should fail!"
            except AssertionError as e:
                pass

        array1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        array2 = np.array([[1, 2], [5, 6], [3, 4], [7, 8]])
        __assert_equal_unordered(array1, array2)

        # more fuzz testing with passing cases
        for _ in range(10000):
            array1 = np.random.rand(4, 2)
            array2 = array1.copy()
            np.random.shuffle(array2)
            __assert_equal_unordered(array1, array2)

    test___assert_equal_unordered()
