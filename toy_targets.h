//****TOY TARGETS****
// some are also useful as priors or as approximations to targets

// Targets that return 0
double l_null(const Eigen::ArrayXd &x, const Eigen::ArrayXd &thetas);
Eigen::VectorXd gl_null(const Eigen::ArrayXd &x, const Eigen::ArrayXd &thetas);
double gldote_null(const Eigen::ArrayXd &x, const Eigen::VectorXd &e, const Eigen::ArrayXd &thetas);

// ISOTROPIC
double l_GaussIso(const Eigen::ArrayXd &x, const Eigen::ArrayXd &thetas);
Eigen::VectorXd gl_GaussIso(const Eigen::ArrayXd &x, const Eigen::ArrayXd &thetas);
double gldote_GaussIso(const Eigen::ArrayXd &x, const Eigen::VectorXd &e, const Eigen::ArrayXd &thetas);
double l_PowIso(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
Eigen::VectorXd gl_PowIso(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
double gldote_PowIso(const Eigen::ArrayXd &x, const Eigen::VectorXd &e, const Eigen::ArrayXd &theta);

//**********MORE GENERAL GAUSSIAN************
// theta = [E1...Ed,V1...Vd]
double l_GaussDiag(const Eigen::ArrayXd &x, const Eigen::ArrayXd &th);
Eigen::VectorXd gl_GaussDiag(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
double gldote_GaussDiag(const Eigen::ArrayXd &x, const Eigen::VectorXd &e, const Eigen::ArrayXd &theta);
double l_GaussLinSD(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
Eigen::VectorXd gl_GaussLinSD(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
double gldote_GaussLinSD(const Eigen::ArrayXd &x, const Eigen::VectorXd &e, const Eigen::ArrayXd &theta);
double l_GaussLinH(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
Eigen::VectorXd gl_GaussLinH(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
double gldote_GaussLinH(const Eigen::ArrayXd &x, const Eigen::VectorXd &e, const Eigen::ArrayXd &theta);

//*************** MORE GENERAL EXPONENTIAL POWER LAW **************
double l_PowLinSc(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
Eigen::VectorXd gl_PowLinSc(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
double gldote_PowLinSc(const Eigen::ArrayXd &x, const Eigen::VectorXd &e, const Eigen::ArrayXd &theta);
double l_PowProdLinSc(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
Eigen::VectorXd gl_PowProdLinSc(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
double gldote_PowProdLinSc(const Eigen::ArrayXd &x, const Eigen::VectorXd &e, const Eigen::ArrayXd &theta);

// ************** Logistic ******************
//theta = [E1...Ed,Sc1...Scd]
double l_LogisticCpts(const Eigen::ArrayXd &x, const Eigen::ArrayXd &th); 
Eigen::VectorXd gl_LogisticCpts(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);

//*************** Student-t ******************
double l_MVtDiagLinSc(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
Eigen::VectorXd gl_MVtDiagLinSc(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
double gldote_MVtDiagLinSc(const Eigen::ArrayXd &x, const Eigen::VectorXd &e, const Eigen::ArrayXd &theta);
double l_tProdLinSc(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
Eigen::VectorXd gl_tProdLinSc(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);
double gldote_tProdLinSc(const Eigen::ArrayXd &x, const Eigen::VectorXd &e, const Eigen::ArrayXd &theta);
double l_MVt(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);

// ************** SkewNormal ******************
//theta = [alpha,Sc1...Scd]
double l_SkewNormalCpts(const Eigen::ArrayXd &x, const Eigen::ArrayXd &th); 
Eigen::VectorXd gl_SkewNormalCpts(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);

// ************** Modified Rosenbrock ************
double l_ModifiedRosenbrock(const Eigen::ArrayXd &x, const Eigen::ArrayXd &th); 
Eigen::VectorXd gl_ModifiedRosenbrock(const Eigen::ArrayXd &x, const Eigen::ArrayXd &theta);


// ******NUMERICAL DIFFERENTIATION*******
Eigen::VectorXd glp_from_lp(const Eigen::ArrayXd &xs, const Eigen::ArrayXd &theta, double (*plp_fn)(const Eigen::ArrayXd &xs, const Eigen::ArrayXd &thetas), const double eps=1e-5);
Eigen::MatrixXd Hlp_from_lp(const Eigen::ArrayXd &xs, const Eigen::ArrayXd &theta, double (*plp_fn)(const Eigen::ArrayXd &xs, const Eigen::ArrayXd &theta), const double eps=1e-5);

// **** Std Normal CDF ****
double Phi(const double x);
double logPhi(const double x);
double phiOPhi(const double x);
Eigen::ArrayXd Phi(const Eigen::ArrayXd &xs);
Eigen::ArrayXd logPhi(const Eigen::ArrayXd &xs);
Eigen::ArrayXd phiOPhi(const Eigen::ArrayXd &xs);

//******* Random samples ********
Eigen::ArrayXd StdNormalAXd(size_t n, std::mt19937 &gen, std::normal_distribution<double> StdNormal);
Eigen::ArrayXd StdSkewNormalAXd(size_t n, double omega, std::mt19937 &gen, std::normal_distribution<double> StdNormal);
Eigen::ArrayXd StdUnifAXd(size_t n, std::mt19937 &gen, std::uniform_real_distribution<double> StdUnif);
Eigen::ArrayXd StdLogisticAXd(size_t n, std::mt19937 &gen, std::uniform_real_distribution<double> StdUnif);
Eigen::ArrayXd ModifiedRosenbrockAXd(size_t n, std::mt19937 &gen, std::normal_distribution<double> StdNormal, const Eigen::ArrayXd &theta);

