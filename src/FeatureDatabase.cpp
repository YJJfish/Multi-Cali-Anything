#include "FeatureDatabase.hpp"
#include "defines.hpp"
#include <iostream>

bool FeatureDatabase::connect(const std::filesystem::path& path) {
	// Connect to the database.
	if (sqlite3_open(path.string().c_str(), &this->_connection) != SQLITE_OK) {
		std::cerr << "[FeatureDatabase] Failed to open the database." << std::endl;
		this->_connection = nullptr;
		this->_stmt = nullptr;
		return false;
	}
	// Prepare the statement.
	const char* query = R"(
		SELECT corner_x, corner_y, dense_feature
		FROM dense_features
		WHERE frame_name = ? AND camera_name = ? AND keypoint_id = ?
	)";
	if (sqlite3_prepare_v2(this->_connection, query, -1, &this->_stmt, nullptr) != SQLITE_OK) {
		std::cerr << "[FeatureDatabase] Failed to prepare the statement." << std::endl;
		this->_connection = nullptr;
		this->_stmt = nullptr;
		return false;
	}
	return true;
}

void FeatureDatabase::disconnect(void) {
	if (this->isConnected()) {
		sqlite3_finalize(this->_stmt);
		sqlite3_close(this->_connection);
	}
	this->_connection = nullptr;
	this->_stmt = nullptr;
}

bool FeatureDatabase::loadFeaturePatch(std::uint64_t frameName, std::uint64_t cameraName, std::uint64_t keypointID, FeaturePatch& featurePatch) const {
	// Use the reference version.
	FeaturePatchRef featurePatchRef{};
	if (!this->loadFeaturePatch(frameName, cameraName, keypointID, featurePatchRef))
		return false;
	// Copy the data.
	featurePatch.frameName = frameName;
	featurePatch.cameraName = cameraName;
	featurePatch.keypointID = keypointID;
	featurePatch.cornerX = featurePatchRef.cornerX;
	featurePatch.cornerY = featurePatchRef.cornerY;
	featurePatch.data = featurePatchRef.data;
	return true;
}

bool FeatureDatabase::loadFeaturePatch(std::uint64_t frameName, std::uint64_t cameraName, std::uint64_t keypointID, FeaturePatchRef& featurePatchRef) const {
	if (!this->isConnected()) {
		std::cerr << "[FeatureDatabase] Database is not connected." << std::endl;
		return false;
	}
	// Execute the query.
	sqlite3_reset(this->_stmt);
	sqlite3_bind_int64(_stmt, 1, static_cast<sqlite3_int64>(frameName));
	sqlite3_bind_int64(_stmt, 2, static_cast<sqlite3_int64>(cameraName));
	sqlite3_bind_int64(_stmt, 3, static_cast<sqlite3_int64>(keypointID));
	// There should be exactly one row (having the feature patch) or zero row (no feature patch).
	if (sqlite3_step(_stmt) == SQLITE_ROW) {
		std::uint64_t cornerX = static_cast<std::uint64_t>(sqlite3_column_int64(_stmt, 0));
		std::uint64_t cornerY = static_cast<std::uint64_t>(sqlite3_column_int64(_stmt, 1));
		const void* denseFeature = sqlite3_column_blob(_stmt, 2);
		std::size_t denseFeatureSize = static_cast<std::size_t>(sqlite3_column_bytes(_stmt, 2));
		// Validate the dense feature size.
		if (denseFeatureSize != FEATURE_PATCH_WIDTH * FEATURE_PATCH_HEIGHT * FEATURE_PATCH_CHANNEL * sizeof(Eigen::half)) {
			// Invalid dense feature size.
			std::cerr << "[FeatureDatabase] Invalid dense feature size for frame " << frameName << ", camera " << cameraName << ", keypoint " << keypointID << "." << std::endl;
			std::cerr << "[FeatureDatabase] Expected: " << FEATURE_PATCH_WIDTH * FEATURE_PATCH_HEIGHT * FEATURE_PATCH_CHANNEL * sizeof(Eigen::half) << std::endl;
			std::cerr << "[FeatureDatabase] Actual: " << denseFeatureSize << std::endl;
			return false;
		}
		// Return the feature patch reference.
		featurePatchRef.frameName = frameName;
		featurePatchRef.cameraName = cameraName;
		featurePatchRef.keypointID = keypointID;
		featurePatchRef.cornerX = cornerX;
		featurePatchRef.cornerY = cornerY;
		new (&featurePatchRef.data) FeaturePatchRef::DataRef(
			reinterpret_cast<const Eigen::half*>(denseFeature),
			FEATURE_PATCH_WIDTH * FEATURE_PATCH_HEIGHT,
			FEATURE_PATCH_CHANNEL
		);
		return true;
	} else {
		// No feature found.
		std::cerr << "[FeatureDatabase] No feature found for frame " << frameName << ", camera " << cameraName << ", keypoint " << keypointID << "." << std::endl;
		return false;
	}
}