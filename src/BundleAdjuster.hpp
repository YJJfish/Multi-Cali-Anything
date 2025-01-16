#include "Reconstruction.hpp"
#include "KRT.hpp"
#include "FeatureDatabase.hpp"

/****************************************************************************************
 * @struct BundleAdjusterParameters
 * @brief Parameters for the bundle adjuster.
 ****************************************************************************************/
struct BundleAdjusterParameters {

	double lambda0 = 1.00; ///< The coefficient for reprojection loss.
	double lambda1 = 0.01; ///< The initial coefficient for rotation extrinsics regularization loss.
	double lambda2 = 0.01; ///< The initial coefficient for translation extrinsics regularization loss.
	double lambda3 = 0.01; ///< The coefficient for featuremetric reprojection loss.
	double lambda4 = 0.02; ///< The initial coefficient for focal length intrinsics variance loss.
	double lambda5 = 0.02; ///< The initial coefficient for principal point intrinsics variance loss.
	double scale = 2.0; ///< The scale factor to be applied with lambda1, lambda2, and lambda4.
	double terminationLambda1 = 1e6; ///< The termination value for lambda1.

	/**
	  * @brief Load the parameters from the given path.
	  * @param path 	The path to the input file.
	  * @return True if the parameters are loaded successfully.
	  */
	bool load(const std::filesystem::path& path);

	/**
	  * @brief Save the parameters to the given path.
	  * @param path 	The path to the output text file.
	  * @return True if the parameters are saved successfully.
	  */
	bool save(const std::filesystem::path& path) const;

};

/****************************************************************************************
 * @class BundleAdjuster
 * @brief Class to perform bundle adjustment optimization.
 ****************************************************************************************/
class BundleAdjuster {

public:

	/** 
	  * @brief Default constructor.
	  */
	BundleAdjuster(void) = default;
	
	/** 
	  * @brief Perform the optimization.
	  * 
	  * Make sure that:
	  *  - pReconstructions is set and all reconstructions are initialized.
	  *  - pKRT is set.
	  *  - pFeatureDatabase is set and connected.
	  * After the optimization, the results are stored as:
	  *  - pReconstructions are updated as the optimized reconstructions.
	  *  - globalIntrinsics stores the optimized global intrinsics.
	  */
	void optimize(void);

	/**
	  * @brief Save the history of costs and errors to the given path.
	  * @param path 	The path to the output CSV file.
	  * @return True if the history is saved successfully.
	  * 
	  * The first line is the header: Name,Value0,Value1,Value2,...
	  * The following lines are the history of costs and errors.
	  */
	bool saveHistory(const std::filesystem::path& path) const;

	/**
	  * @brief Save the global intrinsics to the given path.
	  * @param path 	The path to the output text file.
	  * @param krt 		The KRT data. This is used to get the extrinsics.
	  * @return True if the global intrinsics are saved successfully.
	  * 
	  * The file format is the same as the KRT file.
	  */
	bool saveGlobalIntrinsics(const std::filesystem::path& path, const KRT& krt) const;

public:

	/**
	  * @brief The parameters for the bundle adjuster.
	  */
	BundleAdjusterParameters parameters{};

	/**
	  * @brief Pointer to the reconstructions.
	  */
	std::map<std::uint64_t, Reconstruction>* pReconstructions = nullptr;

	/**
	  * @brief Pointer to the KRT.
	  */
	const KRT* pKRT = nullptr;

	/**
	  * @brief Pointer to the feature database.
	  */
	const FeatureDatabase* pFeatureDatabase = nullptr;

	/**
	  * @brief A map storing the global intrinsics.
	  */
	std::map<std::uint64_t, Eigen::Vector4d> globalIntrinsics{};

	/**
	  * @brief The history of costs and errors during the optimization.
	  */
	std::vector<double> reprojectionCostHistory{};
	std::vector<double> featuremetricReprojectionCostHistory{};
	std::vector<double> rotationRegularizationCostHistory{};
	std::vector<double> translationRegularizationCostHistory{};
	std::vector<double> focalLengthVarianceCostHistory{};
	std::vector<double> principalPointVarianceCostHistory{};
	std::vector<double> focalLengthAbsErrorHistory{};
	std::vector<double> focalLengthRelErrorHistory{};
	std::vector<double> principalPointAbsErrorHistory{};
	std::vector<double> principalPointRelErrorHistory{};

private:


};