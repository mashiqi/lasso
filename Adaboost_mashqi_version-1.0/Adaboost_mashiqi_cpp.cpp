#include "mex.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <math.h>

using std::vector;

// macro definition
#define XX(i,j) X_Pointer[(i)+(j)*nSample]
#define YY(i) y_Pointer[(i)]

// construct the tree structure
typedef struct tree{
	int 	index; 		// indicate making stump by which index
	double 	threshold; 	// threshold for making hyperplane
	double 	direction; 	// We mark these entries that less than threshold as on the left branch, otherwise are on the right branch.
						// If(direction==-1), then the entries on the left  are predicted -1, right are 1;
						// If(direction== 1), then the entries on the right are predicted -1, left  are 1;
	double 	leftIsLeaf, rightIsLeaf; // 0:non-leaf, 1:leaf
	double 	leftPrediction, rightPrediction;	// if(is leaf), there will be some prediction value
	struct tree *nextLeftTree, *nextRightTree; 	// if(isn't leaf), there will be pointers to subtree
} treeStruct;

// construct the output structure for the Adaboost function
typedef struct AdaboostOutput{
	vector<treeStruct> tree; // the regression tree
	vector<double> beta; // the coefficient of every Adaboost weak tree
	vector< vector<double> > weight; // every nth column of variable 'weight' is a weight vector for the nth tree
	vector<double> trainingError; // the training error
	vector<double> exponentialLoss; // the exponential loss
} AdaboostOutputStruct;


// function declaration
AdaboostOutputStruct Adaboost_mashiqi(	vector< vector<double> >	X,
										vector<double> 				y,
										int 						nSample,
										int 						nPredictor,
										int 						stairNumber,
										int 						treeNumber,
										double 						epsTolerance,
										bool 						printInfo,
										bool 						printFigure,
										double 						stopReason);

vector<double> RegressFunction(			vector<double> 				x,
										double 						threshold,
										double 						direction);

										
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
/*
//  Adaboost algorithm, MATLAB/C++ interface 'mexFunction' function.
//
//
//  AUTHOR          - Shiqi Ma (mashiqi01@gmail.com, http://mashiqi.github.io/)
//  DATE            - 02/07/2015
//  VERSION         - 1.0
// 
// 
// There should be some instructions. Coming soon!
// 
// 
// INPUT ARGUMENTS:
// 
//  nlhs    the arguments number of inputs
// 
//  nrhs    the arguments number of outputs
// 
//  prhs[0] X
// 
//  prhs[1] y
// 
//  prhs[2] options
//              |---.(int)maxIteration    - the maximum number of algorithm iterations
//              |---.(int)stairNumber     - stair number
//              |---.(int)treeNumber      - maximum tree number
//              |---.(double)epsTolerance - the tolerance threshold
//              |---.(bool)printInfo      - flag indicating whether to show algorithm's detail information
//              |---.(bool)printFigure    - flag indicating whether to show figures
//              |---.(int)stopReason      - algorithm stop reasons:
//                                              0 - initial value;
//                                              1 - convergence reached;
//                                              2 - maxIteration reached;
// 
// 
// OUTPUT ARGUMENTS:
// 
//  plhs[0] (treeStruct)tree                             // the regression tree	
//                       |---.(vector<double>)index;     // indicate making stump by which index
//                       |---.(vector<double>)threshold; // threshold for making hyperplane
//                       |---.(vector<double>)direction; // If(direction== 1(-1)), the entries on the left are predicted as -1(1), right are 1(-1)
//                       |---.(vector<double>)leftIsLeaf, rightIsLeaf;            // 0:non-leaf, 1:leaf
//                       |---.(vector<double>)leftPrediction, rightPrediction;    // if(is leaf), there will be some prediction value
//                       |---.(vector<treeStruct>)*nextLeftTree, *nextRightTree;  // if(isn't leaf), there will be pointers to subtree
// 
//  plhs[1] (vector<double>)beta                         // the coefficient of every Adaboost weak tree
// 
//  plhs[2] (vector< vector<double> >)weight             // every nth column of variable 'weight' is a weight vector for the nth tree
// 
//  plhs[3] (vector<double>)trainingError                // the training error
// 
//  plhs[4] (vector<double>)exponentialLoss              // the exponential loss
// 
// 
// EXAMPLE:
// 
// This function do not have an example.
// 
// 
// REFERENCE:
// 
//  [1] http://www.mathworks.com/matlabcentral/fileexchange/42130-boosted-binary-regression-trees
*/
{
	// ------------------------ Check for proper number of input and output arguments ---------------- //
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("struct:nrhs", "Three inputs required.");
    } else if (nlhs != 5) {
        mexErrMsgIdAndTxt("struct:nlhs", "Five outputs required.");
    } else if (!mxIsStruct(prhs[2])) {
        mexErrMsgIdAndTxt("struct:wrongType", "Third input must be a structure.");
    }
	
	// ------------------------ Convert arguments to a C/C++ friendly format  ------------------------ //
	// get the dimensional information
	int nSample    = mxGetM( prhs[0] );
	int nPredictor = mxGetN( prhs[0] );
	
	// convert X and y to double type
	double* X_Pointer = (double*)mxGetData( prhs[0] );
	double* y_Pointer = (double*)mxGetData( prhs[1] );
	vector< vector<double> > 	X(nSample);
	vector<double> 				y(nSample);
	for( int iSample = 0; iSample < nSample; iSample++ ) {
		vector<double>			iRow(nPredictor);
		for( int iPredictor = 0; iPredictor < nPredictor; iPredictor++ ) {
			iRow[iPredictor] = 	XX(iSample,iPredictor);
		}
		X[iSample] = iRow;
		y[iSample] = YY(iSample);
	}
    
	// stair number
	double *pStairNumber; pStairNumber = (double*)mxGetData(mxGetField( prhs[2], 0, "stairNumber"));
	int stairNumber = (int)pStairNumber[0]; pStairNumber = NULL;
	if(stairNumber <= 1) {
		mexPrintf("%s%d\n", "stairNumber: ", stairNumber);
		mexErrMsgTxt("stair number should greater than 1!"); 
	}
    
	// maximum tree number
	double *pTreeNumber; pTreeNumber = (double*)mxGetData(mxGetField( prhs[2], 0, "treeNumber"));
	int treeNumber = (int)pTreeNumber[0]; pTreeNumber = NULL;
    
	// the tolerance threshold
	double *pEpsTolerance; pEpsTolerance = (double*)mxGetData(mxGetField( prhs[2], 0, "epsTolerance"));
	double epsTolerance = pEpsTolerance[0]; pEpsTolerance = NULL;

	// flag indicating whether to show algorithm's detail information
	bool *pPrintInfo; pPrintInfo = (bool*)mxGetData(mxGetField( prhs[2], 0, "printInfo"));
	bool printInfo = pPrintInfo[0]; pPrintInfo = NULL;

	// flag indicating whether to show figures
	bool *pPrintFigure; pPrintFigure = (bool*)mxGetData(mxGetField( prhs[2], 0, "printFigure"));
	bool printFigure = pPrintFigure[0]; pPrintFigure = NULL;

	// algorithm stop reasons:
	//       0 - initial value;
	//       1 - convergence reached;
	//       2 - maxIteration reached;
	double *pStopReason; pStopReason = (double*)mxGetData(mxGetField( prhs[2], 0, "stopReason"));
	double stopReason = (int)pStopReason[0]; pStopReason = NULL;
    
    
	// ------------------------ true function applys  ------------------------------------------------ //
	AdaboostOutputStruct adaboostOutput;
	adaboostOutput = Adaboost_mashiqi(  X,
                                        y,
                                        nSample,
                                        nPredictor,
                                        stairNumber,   // stair number
                                        treeNumber,    // maximum tree number
                                        epsTolerance,  // the tolerance threshold
                                        printInfo,     // flag indicating whether to show algorithm's detail information
                                        printFigure,   // flag indicating whether to show figures
                                        stopReason);   // algorithm stop reasons
    
    
    // ------------------------ Convert arguments to a MATLAB friendly format ------------------------ //
	/////////////////// for output arguments [1]: tree ///////////////////
	vector<treeStruct> tempTreeVector(adaboostOutput.tree);
	const char **fnames;       /* pointers to field names */
	const char *fieldname = "index";
	mxArray *pMxArray;
	int countOfField = 0;
	
	fnames = (const char **) mxCalloc(1, sizeof(*fnames));
	fnames = &fieldname;
	plhs[0] = mxCreateStructMatrix(1, 1, 1, fnames);
    mxFree((void *)fnames);
	
	// for tree.index
	pMxArray = mxCreateNumericMatrix(1, treeNumber, mxINT32_CLASS, mxREAL);
	int *pTreeIndex = (int*)mxGetData(pMxArray);
	for ( int iTree = 0; iTree < treeNumber; iTree++ ) {
		pTreeIndex[iTree] = (tempTreeVector[iTree]).index + 1;
	}
	mxSetFieldByNumber( plhs[0], 0, countOfField++/*field: index*/, pMxArray);
	
	// for tree.threshold
	pMxArray = mxCreateNumericMatrix(1, treeNumber, mxDOUBLE_CLASS, mxREAL);
	double *pTreeThreshold = mxGetPr(pMxArray);
	for ( int iTree = 0; iTree < treeNumber; iTree++ ) {
		pTreeThreshold[iTree] = (tempTreeVector[iTree]).threshold;
	}
	fieldname = "threshold";
	mxAddField(plhs[0], fieldname); // add field name
	mxSetFieldByNumber( plhs[0], 0, countOfField++/*field: threshold*/, pMxArray);
	
	// for tree.direction
	pMxArray = mxCreateNumericMatrix(1, treeNumber, mxDOUBLE_CLASS, mxREAL);
	double *pTreeDirection = mxGetPr(pMxArray);
	for ( int iTree = 0; iTree < treeNumber; iTree++ ) {
		pTreeDirection[iTree] = (tempTreeVector[iTree]).direction;
	}
	fieldname = "direction";
	mxAddField(plhs[0], fieldname); // add field name
	mxSetFieldByNumber( plhs[0], 0, countOfField++/*field: direction*/, pMxArray);
	
	// for tree.leftIsLeaf
	pMxArray = mxCreateNumericMatrix(1, treeNumber, mxDOUBLE_CLASS, mxREAL);
	double *pLeftIsLeaf = mxGetPr(pMxArray);
	for ( int iTree = 0; iTree < treeNumber; iTree++ ) {
		pLeftIsLeaf[iTree] = (tempTreeVector[iTree]).leftIsLeaf;
	}
	fieldname = "leftIsLeaf";
	mxAddField(plhs[0], fieldname); // add field name
	mxSetFieldByNumber( plhs[0], 0, countOfField++/*field: direction*/, pMxArray);
	
	// for tree.rightIsLeaf
	pMxArray = mxCreateNumericMatrix(1, treeNumber, mxDOUBLE_CLASS, mxREAL);
	double *pRightIsLeaf = mxGetPr(pMxArray);
	for ( int iTree = 0; iTree < treeNumber; iTree++ ) {
		pRightIsLeaf[iTree] = (tempTreeVector[iTree]).rightIsLeaf;
	}
	fieldname = "rightIsLeaf";
	mxAddField(plhs[0], fieldname); // add field name
	mxSetFieldByNumber( plhs[0], 0, countOfField++/*field: direction*/, pMxArray);
	
	// for tree.leftPrediction
	pMxArray = mxCreateNumericMatrix(1, treeNumber, mxDOUBLE_CLASS, mxREAL);
	double *pLeftPrediction = mxGetPr(pMxArray);
	for ( int iTree = 0; iTree < treeNumber; iTree++ ) {
		pLeftPrediction[iTree] = (tempTreeVector[iTree]).leftPrediction;
	}
	fieldname = "leftPrediction";
	mxAddField(plhs[0], fieldname); // add field name
	mxSetFieldByNumber( plhs[0], 0, countOfField++/*field: leftPrediction*/, pMxArray);
	
	// for tree.rightPrediction
	pMxArray = mxCreateNumericMatrix(1, treeNumber, mxDOUBLE_CLASS, mxREAL);
	double *pRightPrediction = mxGetPr(pMxArray);
	for ( int iTree = 0; iTree < treeNumber; iTree++ ) {
		pRightPrediction[iTree] = (tempTreeVector[iTree]).rightPrediction;
	}
	fieldname = "rightPrediction";
	mxAddField(plhs[0], fieldname); // add field name
	mxSetFieldByNumber( plhs[0], 0, countOfField++/*field: rightPrediction*/, pMxArray);
	
	// for tree.nextLeftTree
	pMxArray = mxCreateNumericMatrix(1, treeNumber, mxDOUBLE_CLASS, mxREAL);
	treeStruct *pNextLeftTree = (treeStruct*)mxGetData(pMxArray);
	for ( int iTree = 0; iTree < treeNumber; iTree++ ) {
		pNextLeftTree = NULL;
		pNextLeftTree++;
	}
	fieldname = "nextLeftTree";
	mxAddField(plhs[0], fieldname); // add field name
	mxSetFieldByNumber( plhs[0], 0, countOfField++/*field: nextLeftTree*/, pMxArray);
	
	// for tree.nextRightTree
	pMxArray = mxCreateNumericMatrix(1, treeNumber, mxDOUBLE_CLASS, mxREAL);
	treeStruct *pNextRightTree = (treeStruct*)mxGetData(pMxArray);
	for ( int iTree = 0; iTree < treeNumber; iTree++ ) {
		pNextRightTree = NULL;
		pNextRightTree++;
	}
	fieldname = "nextRightTree";
	mxAddField(plhs[0], fieldname); // add field name
	mxSetFieldByNumber( plhs[0], 0, countOfField++/*field: nextRightTree*/, pMxArray);
	
	// for tree.treeNumber
	pMxArray = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
	int *int_pTreeNumber = (int*)mxGetData(pMxArray);
	int_pTreeNumber[0] = treeNumber;
	fieldname = "treeNumber";
	mxAddField(plhs[0], fieldname); // add field name
	mxSetFieldByNumber( plhs[0], 0, countOfField++/*field: treeNumber*/, pMxArray);
	
	
	/////////////////// for output arguments [2]: beta ///////////////////
	plhs[1] = mxCreateNumericMatrix(1, treeNumber, mxDOUBLE_CLASS, mxREAL);
    double *p1 = mxGetPr(plhs[1]);
    for ( int iTree = 0; iTree < treeNumber; iTree++ ) {
        p1[iTree] = adaboostOutput.beta[iTree];
    }
	
	/////////////////// for output arguments [3]: weight ///////////////////
	plhs[2] = mxCreateNumericMatrix(nSample, treeNumber, mxDOUBLE_CLASS, mxREAL);
    double *p2 = mxGetPr(plhs[2]);
	for( int iTree = 0; iTree != treeNumber; iTree++ ) {
		for( int iSample = 0; iSample != nSample; iSample++ ) {
            p2[iSample+iTree*nSample] = adaboostOutput.weight[iTree][iSample];
		}
    }
	
	/////////////////// for output arguments [4]: trainingError ///////////////////
	plhs[3] = mxCreateNumericMatrix(1, treeNumber, mxDOUBLE_CLASS, mxREAL);
    double *p3 = mxGetPr(plhs[3]);
    for ( int iTree = 0; iTree < treeNumber; iTree++ ) {
        p3[iTree] = adaboostOutput.trainingError[iTree];
    }
	
	/////////////////// for output arguments [5]: exponentialLoss ///////////////////
	plhs[4] = mxCreateNumericMatrix(1, treeNumber, mxDOUBLE_CLASS, mxREAL);
    double *p4 = mxGetPr(plhs[4]);
    for ( int iTree = 0; iTree < treeNumber; iTree++ ) {
        p4[iTree] = adaboostOutput.exponentialLoss[iTree];
    }
}


AdaboostOutputStruct Adaboost_mashiqi(	vector< vector<double> >	X,
										vector<double> 				y,
										int 						nSample,
										int 						nPredictor,
										int 						stairNumber,
										int 						treeNumber,
										double 						epsTolerance,
										bool 						printInfo,
										bool 						printFigure,
										double 						stopReason)
/*
//  Adaboost algorithm.
//
//
//  AUTHOR          - Shiqi Ma (mashiqi01@gmail.com, http://mashiqi.github.io/)
//  DATE            - 02/07/2015
//  VERSION         - 1.0
// 
// 
// There should be some instructions. Coming soon!
// 
// 
// INPUT ARGUMENTS:
// 
//  X          		- X
// 
//  y          		- y
// 
//  nSample        	- number of samples
// 
//  nPredictor     	- number of predictors
// 
//  stairNumber    	- stair number
// 
//  treeNumber     	- maximum tree number
// 
//  epsTolerance 	- the tolerance threshold
// 
//  printInfo     	- flag indicating whether to show algorithm's detail information
// 
//  printFigure   	- flag indicating whether to show figures
// 
//  stopReason     	- algorithm stop reasons
// 
// 
// OUTPUT ARGUMENTS:
// 
//  adaboostOutput
//     |---.(vector<treeStruct>)tree 			            // the regression tree
//     |							|---.(int)index; 		// indicate making stump by which index
//     |							|---.(double)threshold; // threshold for making hyperplane
//     |							|---.(double)direction; // We mark these entries that less than threshold as on the left branch, otherwise are on the right branch.
//     |													// If(direction== 1), then the entries on the left are predicted -1, right are  1;
//     |													// If(direction==-1), then the entries on the left are predicted  1, right are -1;
//     |							|---.(double)leftIsLeaf, rightIsLeaf; 				// 0:non-leaf, 1:leaf
//     |							|---.(double)leftPrediction, rightPrediction; 		// if(is leaf), there will be some prediction value
//     |							|---.(treeStruct)*nextLeftTree, *nextRightTree;		// if(isn't leaf), there will be pointers to subtree
//     |
//     |---.(vector<double>)beta 				// the coefficient of every Adaboost weak tree
//     |---.(vector< vector<double> >)weight	// every nth column of variable 'weight' is a weight vector for the nth tree
//     |---.(vector<double>)trainingError 		// the training error
//     |---.(vector<double>)exponentialLoss 	// the exponential loss
// 
// 
// EXAMPLE:
// 
// This function do not have an example.
// 
// 
// REFERENCE:
//
// [1] Hastie, Trevor, et al. The elements of statistical learning. Vol.
//     2. No. 1. New York: Springer, 2009.
*/
{							

    // initialization
	vector<treeStruct> AdaboostTree( treeNumber );
	vector<double> columnOfX(nSample);
	vector<double> beta(treeNumber);
	vector<double> columnOfWeight(nSample);
	vector<double> nPrediction(nSample,0);
	vector<double> prediction(nSample,0);
	vector<double> trainingError( treeNumber, 0);
	vector<double> exponentialLoss( treeNumber, 0);
	double 	errorValOpt;
	vector<double> 	nPredictionOpt(nSample,0);
	int 	indexOpt;
	double 	thresholdOpt;
	double 	directionOpt;
	double 	leftIsLeafOpt, rightIsLeafOpt;
	double 	leftPredictionOpt, rightPredictionOpt;
	treeStruct *nextLeftTreeOpt, *nextRightTreeOpt;
	double rangeMin, rangeMax, stairSize;
	std::vector<double> candidateThreshold(stairNumber);
	double threshold;
	double direction;
	double errorVal;
	double sumWeight;
	
	// initialize the weight matrix
	vector< vector<double> > weight( treeNumber+1 );
	{
		vector<double> iColumn(nSample, 1.0/((double)nSample) );
		weight[0] = iColumn;
	}
	for( int iTree = 1; iTree != treeNumber+1; iTree++ ) {
		vector<double> iColumn(nSample,0);
		weight[iTree] = iColumn;
	}
	AdaboostOutputStruct adaboostOutput;
	adaboostOutput.tree 			= AdaboostTree;
	adaboostOutput.beta 			= beta;
	adaboostOutput.weight 			= weight;
	adaboostOutput.trainingError 	= trainingError;
	adaboostOutput.exponentialLoss 	= exponentialLoss;
	
	// Adaboost algorithm begins
	for( int iTree = 0; iTree != treeNumber; iTree++ ) {
	    
		if(printInfo) {
			mexPrintf("treeNumber = %d \t iTree = %d\n",treeNumber,iTree);
		}
		errorValOpt = DBL_MAX;
		
		// compute the stump
		for( int index = 0; index != nPredictor; index++ ) {
		    
			// extract column of X
			for( int i = 0; i != nSample; i++ ) {
				columnOfX[i] = X[i][index];
			}
			
			// compute the value range of this column of X
			rangeMin = *min_element(columnOfX.begin(),columnOfX.end());
			rangeMax = *max_element(columnOfX.begin(),columnOfX.end());
			stairSize = (rangeMax - rangeMin) / (stairNumber - 1);
			
			// compute the candidate threshold according the stair number and the min/max of x
			for( int i = 0; i != stairNumber; i++ ) {
				candidateThreshold[i] = rangeMin + stairSize*i;
			}
			candidateThreshold[0] -= 0.1;
			candidateThreshold[stairNumber-1] += 0.1;
			
			//make split according to these candidate threshold values
			for(int j = 0; j != stairNumber ; j++) {
				threshold = candidateThreshold[j];
				direction = -1;
				for(; direction <= 1; direction += 2 ) {
					// make prediction
					nPrediction = RegressFunction( columnOfX, threshold, direction );
					
					// now compute training error
					errorVal = 0;
					for( int i = 0; i != nSample; i++ ) {
						if( nPrediction[i] != y[i] )
							errorVal += weight[iTree][i];
					}
					
					// store optimal tree information
					if( errorVal < errorValOpt ) { 
						errorValOpt 		= errorVal;
						nPredictionOpt		= nPrediction;
						indexOpt   			= index;
						thresholdOpt		= threshold;
						directionOpt		= direction;
						leftIsLeafOpt		= 1;
						rightIsLeafOpt		= 1;
						nextLeftTreeOpt		= NULL;
						nextRightTreeOpt	= NULL;
						if(direction == -1) {
							leftPredictionOpt	= -1;
							rightPredictionOpt	=  1;
						}
						else {
					        leftPredictionOpt	=  1;
							rightPredictionOpt	= -1;
						}
					}
				}
			}
		}
		// store the nth tree
		AdaboostTree[iTree].index   		= indexOpt;
		AdaboostTree[iTree].threshold		= thresholdOpt;
		AdaboostTree[iTree].direction		= directionOpt;
		AdaboostTree[iTree].leftIsLeaf		= leftIsLeafOpt;
		AdaboostTree[iTree].rightIsLeaf		= rightIsLeafOpt;
		AdaboostTree[iTree].nextLeftTree	= nextLeftTreeOpt;
		AdaboostTree[iTree].nextRightTree	= nextRightTreeOpt;
		if(directionOpt == -1) {
			AdaboostTree[iTree].leftPrediction	= leftPredictionOpt;
			AdaboostTree[iTree].rightPrediction	=  rightPredictionOpt;
		}
		else {
			AdaboostTree[iTree].leftPrediction	=  leftPredictionOpt;
			AdaboostTree[iTree].rightPrediction	= rightPredictionOpt;
		}

		// update parameters
		// update beta. Please refer to the (10.12) of the reference book.
		beta[iTree] = log((1-errorValOpt) / errorValOpt);

		// update weight. Please refer to the (10.14) of the reference book.
		for(int iSample = 0; iSample != nSample; iSample++ ) {
			if( nPredictionOpt[iSample] != y[iSample] )
				weight[iTree+1][iSample] = weight[iTree][iSample] * exp(beta[iTree]);
			else
				weight[iTree+1][iSample] = weight[iTree][iSample];
		}
		// normalize the weight vector
		sumWeight = 0;
		for(int i = 0; i != nSample; i++ ) {
			sumWeight += weight[iTree+1][i];
		}
		for(int i = 0; i != nSample; i++ ) {
			weight[iTree+1][i] = weight[iTree+1][i]/sumWeight;
		}

		// make prediction
		for(int i = 0; i != nSample; i++ ) {
			prediction[i] += beta[iTree]*nPredictionOpt[i];
		}
		for(int i = 0; i != nSample; i++ ) {
			if( !( (y[i]>0 && prediction[i]>0) || (y[i]<0 && prediction[i]<0) || (y[i]==0 && prediction[i]==0) ) )
				// execute the following line when signs are not equal
				trainingError[iTree]++;
		}
		trainingError[iTree] = trainingError[iTree] / nSample;
		for(int i = 0; i != nSample; i++ ) {
			exponentialLoss[iTree] += exp(-y[i]*prediction[i]);
		}
		exponentialLoss[iTree] = exponentialLoss[iTree] / nSample;
		
		// store results to the output structure
		adaboostOutput.tree[iTree] = AdaboostTree[iTree];
		adaboostOutput.beta[iTree] = beta[iTree];
		for(int i = 0; i != nSample; i++ ) {
			columnOfWeight[i] = weight[iTree][i];
		}
		adaboostOutput.weight[iTree] = columnOfWeight;
		adaboostOutput.trainingError[iTree] = trainingError[iTree];
		adaboostOutput.exponentialLoss[iTree] = exponentialLoss[iTree];
	}
	return adaboostOutput;
}
	

vector<double> RegressFunction( vector<double>  x,
                                double          threshold,
                                double          direction)
/*
//  Regression function for Adaboost algorithm.
//
//
//  AUTHOR          - Shiqi Ma (mashiqi01@gmail.com, http://mashiqi.github.io/)
//  DATE            - 02/07/2015
//  VERSION         - 1.0
// 
// 
// There should be some instructions. Coming soon!
// 
// 
// INPUT ARGUMENTS:
// 
//  x           	- the vertical vector
// 
//  threshold   	- should be a scalar
// 
//  direction      	- should be the same length as "x"
// 
// 
// OUTPUT ARGUMENTS:
// 
//  yHat            - prediction
// 
// 
// EXAMPLE:
// 
// This function do not have an example.
// 
// 
// REFERENCE:
//
// [1] Hastie, Trevor, et al. The elements of statistical learning. Vol.
//     2. No. 1. New York: Springer, 2009.
*/
{
    // initialization
    int nSample = x.size();
	std::vector<double> yHat(nSample,1); // initialize all entries of yHat to 1
	double error_IfRightIs1 = 0, error_IfLeftIs1 = 0;
	
	if(direction == 1) {
		for(int i = 0; i != nSample; i++ ) {
			if(x[i] < threshold)
				yHat[i] = -1;
		}
	}
	else {
	    for(int i = 0; i != nSample; i++ ) {
			if(x[i] > threshold)
				yHat[i] = -1;
		}
	}
	return yHat;
}