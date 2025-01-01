#pragma once
#include <filesystem>
#include <sqlite3.h>
#include <Eigen/Eigen>

/****************************************************************************************
 * @struct FeaturePatch
 * @brief Structure to store a feature patch.
 * 
 * This structure copies and stores the feature patch data loaded from the SQLite database.
 ****************************************************************************************/
struct FeaturePatch {

public:

	std::uint64_t frameName = 0ULL;
	std::uint64_t cameraName = 0ULL;
	std::uint64_t keypointID = 0ULL;
	std::uint64_t cornerX = 0ULL;
	std::uint64_t cornerY = 0ULL;

	using Data = Eigen::Matrix<Eigen::half, Eigen::Dynamic, 128, Eigen::RowMajor>;

	Data data = Data(0, 128);

};

/****************************************************************************************
 * @struct FeaturePatchRef
 * @brief Structure to reference a feature patch.
 * 
 * This structure references the feature patch data loaded from the SQLite database.
 * The data becomes invalid when the database executes another query.
 ****************************************************************************************/
struct FeaturePatchRef {

public:

	std::uint64_t frameName = 0ULL;
	std::uint64_t cameraName = 0ULL;
	std::uint64_t keypointID = 0ULL;
	std::uint64_t cornerX = 0ULL;
	std::uint64_t cornerY = 0ULL;

	using DataRef = Eigen::Map<const Eigen::Matrix<Eigen::half, Eigen::Dynamic, 128, Eigen::RowMajor>>;

	DataRef data = DataRef(nullptr, 0, 128);

};

/****************************************************************************************
 * @class FeatureDatabase
 * @brief Class to load feature patches from a SQLite database.
 * 
 * The database is expected to be created by "extract_dense_features.py".
 ****************************************************************************************/
class FeatureDatabase {
	
public:

	/**
	  * @brief Default constructor.
	  */
	FeatureDatabase(void) = default;

	/**
	  * @brief Construct the feature database from the given path.
	  * @param path 	The path to the feature database.
	  */
	FeatureDatabase(const std::filesystem::path& path) {
		this->connect(path);
	}

	/**
	  * @brief Connect to the feature database and prepare the statement.
	  * @param path 	The path to the feature database.
	  * @return True if the connection is successful.
	  */
	bool connect(const std::filesystem::path& path);

	/**
	  * @brief Check if the database is connected.
	  * @return True if the database is connected.
	  */
	bool isConnected(void) const {
		return this->_connection != nullptr;
	}

	/**
	  * @brief Disconnect from the feature database.
	  */
	void disconnect(void);

	/**
	  * @brief Destructor.
	  */
	~FeatureDatabase(void) {
		this->disconnect();
	}

	/**
	  * @brief Load the feature patch from the database.
	  * @param frameName 	The frame name.
	  * @param cameraName 	The camera name.
	  * @param keypointID 	The keypoint ID.
	  * @param featurePatch 	The feature patch to store the data.
	  * @return True if the feature patch is loaded successfully.
	  * 
	  * Memory copy will be performed.
	  */
	bool loadFeaturePatch(std::uint64_t frameName, std::uint64_t cameraName, std::uint64_t keypointID, FeaturePatch& featurePatch) const;

	/**
	  * @brief Load the feature patch reference from the database.
	  * @param frameName 	The frame name.
	  * @param cameraName 	The camera name.
	  * @param keypointID 	The keypoint ID.
	  * @param featurePatchRef 	The feature patch reference to reference the data.
	  * @return True if the feature patch reference is loaded successfully.
	  * 
	  * No memory copy will be performed. The data becomes invalid when the database executes another query.
	  */
	bool loadFeaturePatch(std::uint64_t frameName, std::uint64_t cameraName, std::uint64_t keypointID, FeaturePatchRef& featurePatchRef) const;

private:

	sqlite3* _connection = nullptr;
	sqlite3_stmt* _stmt = nullptr;
	
};
