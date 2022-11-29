//*****************************************************
// Class for prior and likelihood
//*****************************************************

class Targd {
public:
  Eigen::ArrayXd theta; // hyperparameters
  double l_fn(const Eigen::ArrayXd &xs);
  Eigen::VectorXd gl_fn(const Eigen::ArrayXd &xs);
  double (*pl_fn)(const Eigen::ArrayXd &xs, const Eigen::ArrayXd &theta);
  Eigen::VectorXd (*pgl_fn)(const Eigen::ArrayXd &xs, const Eigen::ArrayXd &theta);  
};


void aaps(const int nits, const Eigen::ArrayXd &x0, const int Wtype,
	  const double eps, const int K, const Eigen::ArrayXd &Mdiag,
	  Targd &prior, Targd &likelihood,
	  const std::string outroot, const std::string outpath,
	  const int thin, const int prt);
