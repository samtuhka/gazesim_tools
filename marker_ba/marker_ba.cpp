<%
cfg['libraries'] = ['ceres']
cfg['include_dirs'] = [
	'/usr/local/include/eigen3'
	]
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++1z', '-g', '-Wno-misleading-indentation']
%>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Geometry>
#include <cmath>
#include <experimental/any>

namespace py = pybind11;

using Eigen::Matrix;
using Eigen::RowMajor;

template<typename T> using Point = Matrix<T, 1, 3, RowMajor>;
template<typename T> using Points = Matrix<T, Eigen::Dynamic, 3, RowMajor>;
template<typename T> using Pixel = Matrix<T, 1, 2, RowMajor>;
template<typename T> using Pixels = Matrix<T, Eigen::Dynamic, 2, RowMajor>;
template<typename T> using Tvec = Matrix<T, 1, 3, RowMajor>;
template<typename T> using Rvec = Matrix<T, 1, 3, RowMajor>;
template<typename T> using Projection = Matrix<T, 3, 3, RowMajor>;


template <typename T>
Point<T> transform_point(const Point<T>& point, const Rvec<T>& rvec, const Tvec<T>& tvec) {
	Point<T> p;
	ceres::AngleAxisRotatePoint(rvec.data(), point.data(), p.data());
	p += tvec;
	return p;
}

template <typename T>
Pixel<T> project_point(const Point<T>& point, const Rvec<T>& rvec,
		const Tvec<T>& tvec, const Projection<T>& camera) {
	auto p = transform_point(point, rvec, tvec);
	return (camera*p.transpose()).colwise().hnormalized();
}


struct PointReprojectionError {
	Pixel<double> observed;
	Projection<double> camera;

	PointReprojectionError(const Pixel<double>& observed, Projection<double>& camera)
		:observed(observed), camera(camera) {}
	
	template<typename T>
	bool operator()(
			const T* const cr,
			const T* const ct,
			const T* const point_,
			T* residuals) const {
		auto projected = project_point(
				Point<T>(point_),
				Rvec<T>(cr),
				Tvec<T>(ct),
				camera.cast<T>().eval());
		auto error = (projected - observed.cast<T>().eval()).eval();
		Eigen::Map<decltype(error)> resid(residuals);
		resid = error;
		return true;
	}
	
	static auto Create(auto&&... params) {
		using Me = PointReprojectionError;
		return new ceres::AutoDiffCostFunction<Me, 2, 3, 3, 3>(new Me(params...));
	}

};
using Frame = std::map<int, Pixels<double>>;
using Frames = std::map<int, Frame>;
using Pose = std::tuple<Rvec<double>, Tvec<double>>;

struct RelativePoseError {
	double ts0, ts1, ts2;

	RelativePoseError(double ts0, double ts1, double ts2): ts0(ts0), ts1(ts1), ts2(ts2) {}
	
	template<typename T>
	bool operator()(
			const T* const r0, const T* const t0,
			const T* const r1, const T* const t1,
			const T* const r2, const T* const t2,
			T* residuals) const {
		Rvec<T> rvec0(r0); Tvec<T> tvec0(t0);
		Rvec<T> rvec1(r1); Tvec<T> tvec1(t1);
		Rvec<T> rvec2(r2); Tvec<T> tvec2(t2);
		auto testPoint = Point<T>::Ones().eval();
		auto p0 = transform_point(testPoint, rvec0, tvec0);
		auto p1 = transform_point(testPoint, rvec1, tvec1);
		auto p2 = transform_point(testPoint, rvec2, tvec2);
		auto dt1 = ts1 - ts0;
		auto dt2 = ts2 - ts1;
		
		auto rspeed1 = ((p1 - p0)/T(dt1)).eval();
		auto tspeed1 = ((tvec1 - tvec0)/T(dt1)).eval();
		auto rspeed2 = ((p2 - p1)/T(dt2)).eval();
		auto tspeed2 = ((tvec2 - tvec1)/T(dt2)).eval();
		auto raccel = ((rspeed2 - rspeed1)/T(dt2)).eval();
		auto taccel = ((tspeed2 - tspeed1)/T(dt2)).eval();
		Eigen::Map<decltype(raccel)> rresid(residuals); rresid = raccel;
		Eigen::Map<decltype(taccel)> tresid(residuals+3); tresid = taccel;
		return true;
	}
	
	static auto Create(auto&&... params) {
		using Me = RelativePoseError;
		return new ceres::AutoDiffCostFunction<Me, 3+3, 3,3, 3,3, 3,3>(new Me(params...));
	}

};


auto marker_point_ba(
		Projection<double> camera,
		Frames frames,
		std::map<int, double> frame_times,
		std::map<int, Pose> camera_poses,
		std::map<int, Points<double>> markers,
		int reference_id,
		bool fix_cameras=false,
		bool fix_features=false
		) {
	auto smooth_camera = !fix_cameras && fix_features;
	ceres::Problem problem;
	int prev = -1;
	int prevprev = -1;
	for(auto& frame : frames) {
		if(!camera_poses.count(frame.first)) continue;
		auto* pose = &camera_poses.at(frame.first);
		double *rvec = std::get<0>(*pose).data();
		double *tvec = std::get<1>(*pose).data();
		
		for(auto& marker : frame.second) {
			if(!markers.count(marker.first)) continue; 
			auto& mpoints = markers.at(marker.first);
			for(auto i=0; i < marker.second.rows(); ++i) {
				auto cost = PointReprojectionError::Create(marker.second.row(i), camera);
				double *points = mpoints.row(i).data();
				problem.AddResidualBlock(cost,
					new ceres::SoftLOneLoss(1.0),
					rvec, tvec, points);
				if(fix_cameras) {
					problem.SetParameterBlockConstant(rvec);
					problem.SetParameterBlockConstant(tvec);
				}

				if(fix_features or marker.first == reference_id) {
					problem.SetParameterBlockConstant(points);
				}
			}
		}
		
		if(prev > 0 && prevprev > 0 && smooth_camera) {
			auto cost = RelativePoseError::Create(
					frame_times[frame.first],
					frame_times[prev],
					frame_times[prevprev]);
			pose = &camera_poses.at(prev);
			double *r1 = std::get<0>(*pose).data();
			double *t1 = std::get<1>(*pose).data();
			pose = &camera_poses.at(prevprev);
			double *r2 = std::get<0>(*pose).data();
			double *t2 = std::get<1>(*pose).data();
			problem.AddResidualBlock(cost, NULL,
				rvec, tvec, r1, t1, r2, t2);
		}
		prevprev = prev;
		prev = frame.first;
	}

	
	ceres::Solver::Options options;
	//options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	//} else {
	//	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	//}
	options.max_num_iterations = 100;
	//options.max_solver_time_in_seconds = 10.0;
	//options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
	return std::make_tuple(camera_poses, markers);
}

PYBIND11_PLUGIN(marker_ba) {
using namespace pybind11::literals;
pybind11::module m("marker_ba", "Marker bundle adjustment");
/*m.def("planarPnP", &planarPnP);*/

m.def("transform_point", &transform_point<double>);
m.def("project_point", &project_point<double>);
m.def("marker_point_ba", &marker_point_ba,
		"camera"_a, "frames"_a, "frame_times"_a, "camera_poses"_a, "markers"_a, "reference_id"_a,
		"fix_cameras"_a=false, "fix_features"_a=false);

/*py::class_<MarkerSfm>(m, "MarkerSfm")
	.def(py::init<cv::Mat, cv::Mat, std::tuple<int, int>, int>())
	.def("addFrame", &MarkerSfm::addFrame)
	.def("execute", &MarkerSfm::execute)
;*/
return m.ptr();
}

