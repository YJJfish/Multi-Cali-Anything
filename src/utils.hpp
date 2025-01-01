#include <Eigen/Eigen>
#include <cmath>
#include <ceres/ceres.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

/**
  * @brief Bicubic interpolation.
  * @tparam T 			The data type for input coordinates and output vector.
  * @tparam FP 			The data type for the input data matrix.
  * @tparam C 			The number of channels.
  * @tparam RowMajor 	Whether the data matrix is in row-major format.
  * @param data 		The data. It is treated as a 2D array with size (width * height) * C.
  * @param width 		The width of the data matrix.
  * @param height 		The height of the data matrix.
  * @param x 			The x coordinate.
  * @param y 			The y coordinate.
  * @return The interpolated vector.
  */
template <class T, class FP, std::uint64_t C, bool RowMajor>
Eigen::Vector<T, static_cast<int>(C)> bicubicInterpolation(
	const FP* data, std::uint64_t width, std::uint64_t height, T x, T y
) {
	// Clamp the coordinates.
	x = ceres::fmax(T(0.0), ceres::fmin(static_cast<T>(width) - T(1.0), x));
	y = ceres::fmax(T(0.0), ceres::fmin(static_cast<T>(height) - T(1.0), y));
	// Compute the corner coordinates.
	std::uint64_t x1{};
	std::uint64_t y1{};
	if constexpr (std::is_same_v<T, double>) {
		x1 = static_cast<std::uint64_t>(std::floor(x));
		y1 = static_cast<std::uint64_t>(std::floor(y));
	} else {
		x1 = static_cast<std::uint64_t>(std::floor(x.a));
		y1 = static_cast<std::uint64_t>(std::floor(y.a));
	}
	x1 = std::max(std::uint64_t(1ULL), std::min(x1, static_cast<std::uint64_t>(width - 3ULL)));
	y1 = std::max(std::uint64_t(1ULL), std::min(y1, static_cast<std::uint64_t>(height - 3ULL)));
	// Compute fractional offsets.
	T dx = x - static_cast<T>(x1);
	T dy = y - static_cast<T>(y1);
	// Cubic interpolation weights.
	auto cubicWeight = [](T t) -> T {
		T a = T(-0.5); // Catmull-Rom parameter.
		t = ceres::abs(t);
		if (t <= T(1.0)) {
			return (a + T(2.0)) * t * t * t - (a + T(3.0)) * t * t + T(1.0);
		} else if (t < T(2.0)) {
			return a * t * t * t - T(5.0) * a * t * t + T(8.0) * a * t - T(4.0) * a;
		} else {
			return T(0.0);
		}
	};
	// Compute weights for the x and y directions.
	T wx[4], wy[4];
	for (int i = -1; i <= 2; ++i) {
		wx[i + 1] = cubicWeight(dx - static_cast<T>(i));
		wy[i + 1] = cubicWeight(dy - static_cast<T>(i));
	}
	// Perform bicubic interpolation.
	Eigen::Vector<T, static_cast<int>(C)> result = Eigen::Vector<T, static_cast<int>(C)>::Zero();
	for (std::int64_t j = -1; j <= 2; ++j) {
		for (std::int64_t i = -1; i <= 2; ++i) {
			std::uint64_t xi = static_cast<std::uint64_t>(static_cast<std::int64_t>(x1) + i);
			std::uint64_t yi = static_cast<std::uint64_t>(static_cast<std::int64_t>(y1) + j);
			for (std::uint64_t c = 0ULL; c < C; ++c)
				if constexpr (RowMajor)
					result(c) += wx[i + 1] * wy[j + 1] * static_cast<T>(data[(yi * width + xi) * C + c]);
				else
					result(c) += wx[i + 1] * wy[j + 1] * static_cast<T>(data[c * width * height + yi * width + xi]);
		}
	}
	// Return the interpolated feature vector.
	return result;
}

#ifdef __CUDACC__
/**
  * @brief Bicubic interpolation, for CUDA kernels.
  * @tparam T 			The data type for input coordinates and output vector.
  * @tparam FP 			The data type for the input data matrix.
  * @tparam C 			The number of channels.
  * @tparam RowMajor 	Whether the data matrix is in row-major format.
  * @param data 		The data. It is treated as a 2D array with size (width * height) * C.
  * @param width 		The width of the data matrix.
  * @param height 		The height of the data matrix.
  * @param x 			The x coordinate.
  * @param y 			The y coordinate.
  * @return The interpolated vector.
  */
template <class T, class FP, std::uint64_t C, bool RowMajor>
__device__ Eigen::Vector<T, static_cast<int>(C)> bicubicInterpolationDevice(
	const FP* data, std::uint64_t width, std::uint64_t height, T x, T y
) {
	// Clamp the coordinates.
	x = max(T(0.0), min(static_cast<T>(width) - T(1.0), x));
	y = max(T(0.0), min(static_cast<T>(height) - T(1.0), y));
	// Compute the corner coordinates.
	std::uint64_t x1{};
	std::uint64_t y1{};
	x1 = static_cast<std::uint64_t>(floor(x));
	y1 = static_cast<std::uint64_t>(floor(y));
	x1 = max(std::uint64_t(1ULL), min(x1, static_cast<std::uint64_t>(width - 3ULL)));
	y1 = max(std::uint64_t(1ULL), min(y1, static_cast<std::uint64_t>(height - 3ULL)));
	// Compute fractional offsets.
	T dx = x - static_cast<T>(x1);
	T dy = y - static_cast<T>(y1);
	// Cubic interpolation weights.
	auto cubicWeight = [](T t) -> T {
		T a = T(-0.5); // Catmull-Rom parameter.
		t = abs(t);
		if (t <= T(1.0)) {
			return (a + T(2.0)) * t * t * t - (a + T(3.0)) * t * t + T(1.0);
		} else if (t < T(2.0)) {
			return a * t * t * t - T(5.0) * a * t * t + T(8.0) * a * t - T(4.0) * a;
		} else {
			return T(0.0);
		}
	};
	// Compute weights for the x and y directions.
	T wx[4], wy[4];
	for (int i = -1; i <= 2; ++i) {
		wx[i + 1] = cubicWeight(dx - static_cast<T>(i));
		wy[i + 1] = cubicWeight(dy - static_cast<T>(i));
	}
	// Perform bicubic interpolation.
	Eigen::Vector<T, static_cast<int>(C)> result = Eigen::Vector<T, static_cast<int>(C)>::Zero();
	for (std::int64_t j = -1; j <= 2; ++j) {
		for (std::int64_t i = -1; i <= 2; ++i) {
			std::uint64_t xi = static_cast<std::uint64_t>(static_cast<std::int64_t>(x1) + i);
			std::uint64_t yi = static_cast<std::uint64_t>(static_cast<std::int64_t>(y1) + j);
			for (std::uint64_t c = 0ULL; c < C; ++c)
				if constexpr (RowMajor)
					result(c) += wx[i + 1] * wy[j + 1] * static_cast<T>(data[(yi * width + xi) * C + c]);
				else
					result(c) += wx[i + 1] * wy[j + 1] * static_cast<T>(data[c * width * height + yi * width + xi]);
		}
	}
	// Return the interpolated feature vector.
	return result;
}
#endif