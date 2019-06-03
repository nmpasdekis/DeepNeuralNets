#include <Eigen/dense>

inline Eigen::Block<Eigen::MatrixXf, -1, -1, false> outPart(Eigen::MatrixXf & m) {
	return m.block(0, 0, m.rows() - 1, m.cols());
}

inline Eigen::Map<Eigen::MatrixXf> Map(float * data, int Rows, int Cols) {
	return Eigen::Map< Eigen::MatrixXf>(data, Rows, Cols);
}

inline int CorrectMat(Eigen::MatrixXf& m) {
	auto sz = m.size();
	auto* dt = m.data();
	int cnt = 0;
	for (auto i = 0; i<sz; i++) {
		auto& d = dt[i];
		if (!(d>0) && !(d<=0)) {
			cnt++;
			d = 0;
		}
	}
	return cnt;
}