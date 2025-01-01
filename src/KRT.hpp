#pragma once
#include <filesystem>
#include <map>
#include <cstdint>
#include <Eigen/Eigen>

/****************************************************************************************
 * @class KRT
 * @brief Data structure to store the KRT provided by the multiface dataset.
 ****************************************************************************************/
struct KRT {

public:

	/**
	  * @brief Default constructor.
	  */
	KRT(void) = default;

	/**
	  * @brief Load the KRT data from the given file.
	  *
	  * @param path 	The path to the KRT file.
	  * @return True if the KRT data is loaded successfully.
	  */
	bool load(const std::filesystem::path& path);

	/**
	  * @brief Get a string summary of the KRT data.
	  * @return A string summary of the KRT data.
	  */
	std::string summary(void) const;

public:

	/**
	  * @struct CameraParameters
	  * @brief Camera parameters including rotation, translation, and intrinsics.
	  */
	struct CameraParameters {
		Eigen::Vector3d rotation; ///< Rotation in axis-angle representation.
		Eigen::Vector3d translation;
		Eigen::Vector4d intrinsics; ///< fx, fy, cx, cy
	};

	/**
	  * @brief A map storing all camera parameters.
	  *
	  * Key: Camera name.
	  * Value: Camera parameters.
	  */
	std::map<std::uint64_t, CameraParameters> cameraParametersMap;
	
};