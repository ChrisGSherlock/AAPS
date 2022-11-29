// 17/12/2021 CS: Much tidied/rewritten vn of the code 
//                Still needs further tidying e.g. text output at the end. 
//   11/2022 CS: Added Skewed Gaussian and Modified Rosenbrock
// Seperated algorithm code from distribution etc.

// Tested on 10d Gaussian with SDs from 1 to 5 using 10^5 iterations:
//   SDs accurate to < 0.02 for algorithms: 1, 2, 3 (K=4, eps=1.2)
//   and all QQ plots look perfect.
//   Using (K=0, eps=.4) and 10^6 iterations alg 1,2 3 accurate to <.02
// Tested using unstable eps=2.0 and is also correct.

#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Eigenvalues> // needed for preconditioning
#include <chrono>
#include <random>
#include "toy_targets.h"
#include "AAPS.h"

const double Delta=100; // Hard-wired. For large epsilon issues

using namespace std;
using namespace Eigen;

// **********************************************************
// Class for state of the system at the start of an iteration
// Essentially x, p, log pi(x), H and grad log pi(x)
// Also a useful storage box when integrating via leapfrogs
// **********************************************************
class State {
public:
  int d;
  int max_xprt;
  ArrayXd x;
  ArrayXd p;
  double lpi;
  double H;
  ArrayXd g;
  State(int d);
  void Print(std::ostream &os);
};

State::State(int _d) {
  d=_d;
  lpi=0;
  max_xprt=6; // max # components for printing to screen
  x=ArrayXd::Zero(d);
  g=ArrayXd::Zero(d);  
}
void State::Print(std::ostream &os) {
  os << "*State: log pi="<<lpi<<endl; 

  if (x.size() <= max_xprt) {
    os << "x="<<x.transpose()<<endl;
  }
  else {
    os << "x(0:"<<max_xprt-1<<")="<<x.matrix().head(max_xprt).transpose()<<endl;
  }
}

//*****************************************************
// Class for recording and printing AAPS diagnostics
//*****************************************************
// There is much more recording in here than necessary!
// We recorded as much as possible so as to be able to understand
// as much as possible about the algorithm
class AAPSd {
  bool acc, trueacc, unstable; 
  int K, ltot, nstored, ngoodstored, kprop;
  double accratio, accbar, trubar, accbarRB, arbar, ltotbar, ltot2bar;
  double kpropbar, kprop2bar, dK, dn, dng;
  double unstablebar;
  ArrayXi Kproptable;  
public:
  AAPSd(int K);
  void update(bool acc, bool trueacc, bool unstable,
	      int ltot, int kprop, double accratio);
  void printCurr(std::ostream &os);
  void printStats(std::ostream &os);
  bool trueAcc() {return trueacc;}
  double accRatio() {return accratio;}
  double kProp() {return kprop;}
};

AAPSd::AAPSd(int _K) {
  // Inputs (just initialised for tidyness)
  acc=false; trueacc=false; unstable=false; 
  ltot=0; accratio=0; kprop=0;
  // Counters
  nstored=0; ngoodstored=0; dn=0; dng=0; 
  accbar=0; trubar=0; accbarRB=0; arbar=0; unstablebar=0;
  ltotbar=0; ltot2bar=0; kpropbar=0; kprop2bar=0;
  K=_K, dK=(double)K;
  Kproptable=ArrayXi::Zero(K+1);
}

void AAPSd::update(bool _acc, bool _trueacc, bool _unstable,
		   int _ltot, int _kprop, double _accratio) {
  
  acc=_acc; trueacc=_trueacc; unstable=_unstable; ltot=_ltot; kprop=_kprop;
  accratio=_accratio;
  
  nstored++; dn=(double)nstored;
  double dnrec=1.0/dn;

  accbar=(1-dnrec)*accbar+dnrec*acc; // alpha
  trubar=(1-dnrec)*trubar+dnrec*trueacc; // alpha ignoring proposing current
  // Rao-Blackwellised alpha
  accbarRB=(1-dnrec)*accbarRB+dnrec*(accratio<1?accratio:1.0); 
  unstablebar=(1-dnrec)*unstablebar+dnrec*unstable; // frac unstable

  ltotbar=(1-dnrec)*ltotbar+dnrec*ltot;
  ltot2bar=(1-dnrec)*ltot2bar+dnrec*ltot*ltot;

  if (!unstable) { 
    ngoodstored++;
    double dngrec=1/(double)ngoodstored; 

    arbar=(1-dngrec)*arbar+dngrec*accratio;
  
    kpropbar=(1-dngrec)*kpropbar+dngrec*kprop;
    kprop2bar=(1-dngrec)*kprop2bar+dngrec*kprop*kprop;
  
    Kproptable(kprop)++;
  }
}

void AAPSd::printCurr(std::ostream &os) {
  os <<"*AAPSCurr\n";
  os <<"ltot="<<ltot<<", Kprop="<<kprop<<endl;
  os <<"acc="<<acc<<", trueacc="<<trueacc<<", accrat="<<accratio<<endl;
}
void AAPSd::printStats(std::ostream &os) {
  os <<"***APPSStats: nstored="<<nstored<<endl;
  os <<"accbar="<<accbar<<" (RB="<<accbarRB<<"), trubar="<<trubar<<", arbar="<<arbar<<", unstablebar="<<unstablebar<<endl;
  os <<"ltotbar="<<ltotbar<<" (sd="<<sqrt(ltot2bar-ltotbar*ltotbar)<<"), kpropbar="<<kpropbar<<" (sd="<<sqrt(kprop2bar-kpropbar*kpropbar)<<")\n";
  //  os <<"Kprops:  "<<Kproptable.matrix().transpose()<<endl;
  // Normalise by probability density
  ArrayXd Ks=ArrayXd::LinSpaced(K+1,0,K);
  ArrayXd qs=2*(dK+1-Ks)/(dK+1)/(dK+1);
  qs(0)=1/(dK+1);
  ArrayXd Kpropnormed=Kproptable.cast<double>()/qs;
  Kpropnormed=Kpropnormed/Kpropnormed.sum()*100*(dK+1);
  ArrayXi Kpropnormedi=Kpropnormed.cast<int>();
  os <<"Kprops (normalised&standardised): "<<Kpropnormedi.matrix().transpose()<<endl;
}
//*****************************************



//********************** Simple utilities
static void InsertInOutput(State curr, Eigen::ArrayXXd &Output, int &row) {
  int d=curr.d;
  Output.row(row).head(d)=curr.x; // 0 to d-1
  Output.row(row++)(d)=curr.lpi; // d
}

//**************** AAPS ALGORITHM *****************************

// Hamiltonian
double Getlrho(ArrayXd &p, const ArrayXd &Mdiag) {
  return -0.5*(p*p/Mdiag).sum();
}
double Getlpi(ArrayXd &x, Targd &prior, Targd &likelihood) {
  return prior.l_fn(x)+likelihood.l_fn(x);
}
double GetH(ArrayXd &x, ArrayXd &p, const ArrayXd &Mdiag,
	    Targd &prior, Targd &likelihood) {
  double lpi=Getlpi(x, prior, likelihood);
  double lprho=Getlrho(p,Mdiag);
  return -(lpi+lprho);
}

// Summary stats for O(1) memory weighting
double SumWjkL2(ArrayXd xj, double sumpitil, ArrayXd S, ArrayXd S2) {
  double easyBit=sumpitil*(xj.matrix().squaredNorm());
  double sumOther2=S2.sum();
  double sumDots=(xj*S).sum();
  return easyBit-2*sumDots+sumOther2;
}

// Leapfrog through to next apogee
// x0=current position at this iteration of AAPS
// mv=(x,p)
// (x,p): x starts the first call to the function as x0 (both fwd and bwd)
//        p starts as p0 on the first set and -p0 on the second
// (x,p) is altered in place within the function.
// (x,p) on leaving is the point AFTER the end of the segment.
// mvchoose=(xchoose,pchoose)
//         = proposal "at the moment", lchoose its leap number
// store = TRUE => there's a chance of sampling x
// store = FALSE on second call as we don't want to count x0 twice
// l, # points tested, must be initialised externally and is changed in place
// Ditto Spi, Sx and Sx2
void LeapAndChooseToApogee(const double eps, const ArrayXd &Mdiag,
			   const string Wtype, const ArrayXd &x0,
			   const bool try0, 
			   State &mv, bool &unstable,
			   State &mvchoose, int &lchoose,   
			   Targd prior, Targd likelihood,
			   int &lt,int &lu,
			   double &Spi, ArrayXd &Sx, ArrayXd &Sx2,
			   double &Hmax, double &Hmin,
			   mt19937 &gen,uniform_real_distribution<double> Unif,
			   double Hoffset) {
  bool currdotpos=(mv.g*mv.p/Mdiag).sum()>0, hadnegdot=!currdotpos;// 0 counts as -ve
  double wnum, wden;
  bool initial=true; // are we at start of this segment? 
  
  // stop if +ve after having been -ve or unstable
  while (!(currdotpos && hadnegdot) && !unstable) {
    mv.lpi=Getlpi(mv.x,prior,likelihood); // redundant first time each iter
    mv.H=-mv.lpi-Getlrho(mv.p, Mdiag)-Hoffset; //=0 first time each iter
    if ((initial && try0) || (!initial)) { // avoid doubling up on current
      double pitil=exp(-mv.H);
      lu++; // lu tracks points visited that could be sampled from
      Spi+=pitil; // sum of pitildes from points that could be sampled from
      if (Wtype=="pi") {
	wnum=pitil;
	wden=Spi;
      }
      else if (Wtype=="L22") {
	Sx+=mv.x; Sx2+=mv.x*mv.x;
	wnum=(mv.x-x0).matrix().squaredNorm();
	wden=SumWjkL2(x0,(double)lu,Sx,Sx2);
      }
      else if (Wtype=="piL22") {
	Sx+=pitil*mv.x; Sx2+=pitil*mv.x*mv.x;
	wnum=pitil*((mv.x-x0).matrix().squaredNorm());
	wden=SumWjkL2(x0,Spi,Sx,Sx2);
      }

      // Do we choose the new point?
      if (wden==0) { // if just initial point:||x_0-x_0||^2=pi_0||x_0-x_0||^2=0
	mvchoose=mv; lchoose=lt;
      }
      else if (Unif(gen)<wnum/wden) { // guaranteed true first time round
	mvchoose=mv; lchoose=lt;
      }
    }
    
    if (mv.H>Hmax) {
      Hmax=mv.H;
    }
    if (mv.H<Hmin) {
      Hmin=mv.H;
    }

    unstable = (Hmax-Hmin>Delta);

    if (!unstable) { // moves to next point, get x, g but not lpi
      mv.p+=0.5*eps*mv.g;
      mv.x+=eps*mv.p/Mdiag;
      mv.g=(prior.gl_fn(mv.x)+likelihood.gl_fn(mv.x)).array();
      mv.lpi=-1000000; // sanity
      mv.p+=0.5*eps*mv.g;
      lt++; // tracks leapfrogs calculated - for efficiency diagnostic
      currdotpos=(mv.g*mv.p/Mdiag).sum()>0; // 0 counts as -ve
      hadnegdot=hadnegdot || !currdotpos;
    }

    if (initial) initial=false;
  }
}

// F whole segments forward plus the initial fractional segment
// try0=TRUE on 1st call and false on 2nd so don't double count xcurr
bool LeapAndChooseFSegs(const int F, const double eps, const ArrayXd &Mdiag,
			const string Wtype, const bool try0,
			State &mv, int &K,
			Targd &prior, Targd &likelihood, int &lt,int &lu,
			double &Spi, ArrayXd &Sx, ArrayXd &Sx2,
			double &Hmax, double &Hmin,
			mt19937 &gen, uniform_real_distribution<double> Unif,
			double Hoffset) {
  bool unstable=false;
  int i, lchoose,lchooseold, kchoose;
  ArrayXd x0=mv.x, pchoose=mv.p; //x0 = FIXED, initial value for ||x-x0||^2
  State mvchoose(mv.d);

  lt=0; lu=0; Spi=0; Sx=mv.x*0; Sx2=mv.x*0;

  LeapAndChooseToApogee(eps, Mdiag, Wtype, x0, try0, mv,
			unstable, mvchoose, lchoose,
			prior, likelihood, lt, lu,
			Spi, Sx, Sx2, Hmax, Hmin, gen, Unif, Hoffset);
  kchoose=0; // current proposal choice is from segment 0
  lchooseold=lchoose; // current proposal choice's leapfrog number

  for (i=0; i<F; i++) {
    if (!unstable) {
      LeapAndChooseToApogee(eps, Mdiag, Wtype, x0, true, mv,
			    unstable, mvchoose, lchoose,
			    prior, likelihood, lt, lu,
			    Spi, Sx, Sx2, Hmax, Hmin, gen, Unif, Hoffset);
      if (lchoose != lchooseold) { // proposal choice has been updated
	lchooseold=lchoose;
	kchoose=i+1; // now choosing proposal from segment i+1
      }
    }
  }

  if (!unstable) {
    mv=mvchoose; K=kchoose;
  }

  return unstable;
}

// |K|+1 segments will be explored
// Lay K+1-segment "ruler" down uniformly at random
// K=0 is simple single segment sampler
void SampleOnline(const int K, const double eps, const ArrayXd &Mdiag,
		  const string Wtype, State &curr, 
		  Targd &prior, Targd &likelihood, AAPSd &ad,
		  mt19937 &gen, uniform_real_distribution<double> Unif){
  
  double u=Unif(gen), dK=(double)K;
  double Hmax=0, Hmin=0; // for Delta test
  double Hoffset=curr.H; // Added 27/10/2022 for robustness in high d
  int currseg = u*(dK+1-1e-10); // unif between 0 and K
  int Fwd=abs(K)-currseg, Bwd=currseg; 
  int Kfwd, Kbwd, Kprop=-1; // where is proposal from
  int ltfwd=0, ltbwd=0, upfwd=0, upbwd=0; // leapfrog effort, and usable points
  bool accept=false, bogus=false, unstable=false;
  double Spifwd, Spibwd, Wcurrfwd, Wcurrbwd, Wpropfwd, Wpropbwd, rat;
  ArrayXd Sfwd, Sbwd, S2fwd, S2bwd;
  State mvfwd(curr.d), mvbwd(curr.d), prop(curr.d);
  mvfwd=curr; mvbwd=curr; mvbwd.p=-mvbwd.p;
  
  // Leapfrog forwards from current point, sampling as we go
  unstable=LeapAndChooseFSegs(Fwd, eps, Mdiag, Wtype, true,
			      mvfwd, Kfwd, prior, likelihood, 
			      ltfwd, upfwd, Spifwd, Sfwd, S2fwd,
			      Hmax, Hmin, gen, Unif, Hoffset);

  if (!unstable) {
    // Leapfrog backwards from curr, sampling as we go, not sampling curr
    unstable=LeapAndChooseFSegs(Bwd, eps, Mdiag, Wtype, false,
				mvbwd, Kbwd, prior, likelihood, 
				ltbwd, upbwd, Spibwd, Sbwd, S2bwd,
				Hmax, Hmin, gen, Unif, Hoffset);
  }

  if (!unstable) { 
    // Get the \sum part of the numerator in alpha
    if (Wtype=="pi") {
      Wcurrfwd=Spifwd;
      Wcurrbwd=Spibwd;
    }
    else if (Wtype=="L22") {
      Wcurrfwd=SumWjkL2(curr.x,(double)upfwd,Sfwd,S2fwd);
      Wcurrbwd=SumWjkL2(curr.x,(double)upbwd,Sbwd,S2bwd);
    }
    else if (Wtype=="piL22") {
      Wcurrfwd=SumWjkL2(curr.x,Spifwd,Sfwd,S2fwd);
      Wcurrbwd=SumWjkL2(curr.x,Spibwd,Sbwd,S2bwd);
    }

    if (Wcurrfwd+Wcurrbwd==0) { // poss if eps v.large and L2 or piL2
      // the ONLY point in the path is the current point
      prop=mvfwd; Kprop=Kfwd; 
      bogus=true; // propose the current point
    }
    else if (Unif(gen)<Wcurrfwd/(Wcurrfwd+Wcurrbwd)) {
      prop=mvfwd; Kprop=Kfwd; 
      bogus=(mvfwd.x==curr.x).all(); // proposing the current point is really just a reject
    }
    else {
      prop=mvbwd; Kprop=Kbwd; 
    }

    // Get the \sum part of the demoninator in alpha
    if (Wtype=="pi") {
      Wpropfwd=Spifwd;
      Wpropbwd=Spibwd;
    }
    else if (Wtype=="L22") {
      Wpropfwd=SumWjkL2(prop.x,(double)upfwd,Sfwd,S2fwd);
      Wpropbwd=SumWjkL2(prop.x,(double)upbwd,Sbwd,S2bwd);
    }
    else if (Wtype=="piL22") {
      Wpropfwd=SumWjkL2(prop.x,Spifwd,Sfwd,S2fwd);
      Wpropbwd=SumWjkL2(prop.x,Spibwd,Sbwd,S2bwd);
    }
    
    rat=(Wcurrfwd+Wcurrbwd)/(Wpropfwd+Wpropbwd); // equivalent form for sch1-6

    prop.H=prop.H+Hoffset;
    if (Wtype=="L22") {
      rat*=exp(curr.H-prop.H);
    }
    accept=(Unif(gen)<rat);
    if (accept) {
      curr=prop;
    }
  } // all if !unstable

  ad.update(accept,(accept && !bogus), unstable, ltfwd+ltbwd, Kprop, rat);
}


// AAPS
// nits = number of iterations
// x0 = initial vector
// Wtype = weighting scheme
// eps = time step
// K = # additional segments over segment 0
// Mdiag = (inverse preconditioning) vector of masses (diag of matrix)
// prior = prior functions
// likelihood = likelihood functions
// outroot = rootname for output files
// outpath = pathname for output files
// thin = thinning factor (1=no thinning)
// prt = "print every prt iterations"


double Targd::l_fn(const Eigen::ArrayXd &xs) {
  return pl_fn(xs,theta);
}
Eigen::VectorXd Targd::gl_fn(const Eigen::ArrayXd &xs) {
  return pgl_fn(xs,theta);
}

void aaps(const int nits, const ArrayXd &x0, const int Wtype,
	  const double eps, const int K, const ArrayXd &Mdiag,
	  Targd &prior, Targd &likelihood,
	  const string outroot="", const string outpath="./",
	  const int thin=1, const int prt=0) {
  const int d=x0.size(), nout = 1+nits/thin;
  int outrow=0; // stored output row
  ArrayXXd Output(nout,d+1);
  ArrayXd xcurr=x0;
  int i, nacc=0;
  bool precon=((Mdiag-ArrayXd::Constant(d,1.0)).matrix().squaredNorm()>1e-5);
  ArrayXd propSDs=Mdiag.sqrt();
  string outfnameroot="aaps"+outroot+ "_eps" + to_string((int) (eps*100)) + "K"+ to_string(K) + "W" + to_string(Wtype)+"A"+to_string((int) precon);
  string outfname=outfnameroot+".txt";
  string outiname=outfnameroot+".info";
  
  std::random_device rd;  //Will use to obtain seed for random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> Unif(0.0, 1.0);
  std::normal_distribution<double> StdNormal(0.0, 1.0);
  State curr(d);
  curr.x=xcurr;
  curr.lpi=likelihood.l_fn(curr.x)+prior.l_fn(curr.x);
  curr.g=likelihood.gl_fn(curr.x)+prior.gl_fn(curr.x);
  AAPSd ad(K);

  InsertInOutput(curr, Output, outrow);

  auto t1 = std::chrono::high_resolution_clock::now();

  for (i=0;i<nits;i++) {
    
    curr.p=StdNormalAXd(d,gen,StdNormal)*propSDs;
    curr.H=-curr.lpi-Getlrho(curr.p,Mdiag);
    
    if (Wtype==1) { // pi online
      SampleOnline(K, eps, Mdiag, "pi",   curr, prior, likelihood, ad,
		   gen, Unif);
    }
    if (Wtype==2) { // L22 online
      SampleOnline(K, eps, Mdiag, "L22",  curr, prior, likelihood, ad,
		   gen, Unif);
    }
    if (Wtype==3) { // L22_pi online
      SampleOnline(K, eps, Mdiag, "piL22",curr, prior, likelihood, ad,
		   gen, Unif);
    }

    nacc+=ad.trueAcc();
    
    bool toprint=false, tokeep=((i+1)%thin == 0);

    if ((i>0) || (prt>0)) {
      toprint=((i+1) % prt == 0); // account for iteration index starting at 0
    }

    if (toprint){
      ad.printStats(cout);
      curr.Print(cout);
      //      ad.printCurr(cout);
    }
    if (tokeep) { 
      InsertInOutput(curr,Output,outrow);
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();

  ofstream outf, outi;
  outf.open(string(outpath+outfname));  
  outf<<Output;
  outf.close();

  double dnits=(double)nits, dnacc=(double)nacc;
  outi.open(string(outpath+outiname));
  outi <<"\n***";
  outi << "nits=" <<nits<<", eps="<<eps<<", K="<<K<<", precon="<<precon<<"\n";
  outi << "***\nAcc = " << dnacc/dnits<<"\n";
  outi<< "Time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() <<"\n";
  ad.printStats(outi);
  outi.close();
  cout << "outfiles: "<<outfname<<" and "<<outiname<<"\n";
}



  





