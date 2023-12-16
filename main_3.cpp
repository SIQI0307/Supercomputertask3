#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <string>
#include <fstream>
#include <omp.h>
#include <mpi.h>

using namespace std;

#define A_1 -4.
#define A_2 -1.
#define B_1 4.
#define B_2 5.
#define M 160
#define N 160
#define delta 2e-7
#define INDEX(m, i, j) \
	((m)[((i) * N) + (j)])

double h_1 = (B_1 - A_1) / M;
double h_2 = (B_2 - A_2) / N;
double eps = pow(max(h_1, h_2), 2);
int blockM, blockN;

/// initialize all elements of solution(function) to zero as w^0
vector<double> w((M + 1) * (N + 1), 0);

void domainSplit(int &m, int &n, int numProcs)
{
	m = M;
	n = N;
	if (numProcs == 1)
	{
		return;
	}
	int procs = numProcs;
	while (procs > 1)
	{
		if (m >= n)
		{
			m /= 2;
		}
		else
		{
			n /= 2;
		}
		procs /= 2;
	}
}

// F function y=-4/3x+4
double F(double x, double y)
{
	return (((x >= -3) || (x <= 3)) && y >= 0 &&
			((4 * x + 3 * y - 12 <= 0) || (4 * x - 3 * y + 12 >= 0)))
			   ? 1
			   : 0;
}
// some operation in H space, which consists of the "inner" points of a grid function u on rectangle
// scalar product and corresponding norm in grid function space H
double funVScalProd(const vector<double> &u, const vector<double> &v, double h_1, double h_2)
{
	double res = 0;
#pragma omp parallel for reduction(+ : res)
	for (int i = 1; i < M; i++)
	{
		for (int j = 1; j < N; j++)
		{
			res += h_1 * h_2 * u[i * (N + 1) + j] * v[i * (N + 1) + j];
		}
	}
	return res;
}

void funVScalProdMPI(double &res, const vector<double> &u, const vector<double> &v, const int *coords, double h_1, double h_2)
{
	double resP = 0;
	res = 0;
#pragma omp parallel for collapse(2) reduction(+ : resP)
	for (int i = 0; i < blockM + 1; i++)
	{
		for (int j = 0; j < blockN + 1; j++)
		{
			int loc0 = i + coords[0] * blockM, loc1 = j + coords[1] * blockN;
			// int realCoords[2]{ i + coords[0] * blockM,j + coords[1] * blockN };
			// printf("[MPI process %d] Real location at (%d, %d).\n", rank, realCoords[0], loc1);
			/// boundary points won't be calculated, they don't participate in calculation and always equal 0
			if (loc1 == 0 || loc1 == N || loc0 == 0 || loc0 == M)
			{
				continue;
			}
			else if (i == 0 || j == 0)
			{
				continue;
			}
			else
			{
				resP += h_1 * h_2 * u[loc0 * (N + 1) + loc1] * v[loc0 * (N + 1) + loc1];
			}
		}
	}
	MPI_Allreduce(&resP, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// return res;
}
double funVNorm(const vector<double> &u, double h_1, double h_2)
{
	return sqrt(funVScalProd(u, u, h_1, h_2));
}

void funVNormMPI(double &norm, const vector<double> &u, const int *coords, double h_1, double h_2)
{
	funVScalProdMPI(norm, u, u, coords, h_1, h_2);
	norm = sqrt(norm);
}

auto funVSubtract(const vector<double> &u, const vector<double> &v)
{
	vector<double> res((M + 1) * (N + 1), 0);
	// #pragma omp parallel for
	for (int i = 1; i < M; i++)
	{
		for (int j = 1; j < N; j++)
		{
			res[i * (N + 1) + j] = u[i * (N + 1) + j] - v[i * (N + 1) + j];
		}
	}
	return res;
}

void funVSubtractMPI(vector<double> &res, int *coords, const vector<double> &u, const vector<double> &v)
{
	vector<double> resTemp((M + 1) * (N + 1), 0);
// fill(res.begin(), res.end(), 0);
#pragma omp parallel for collapse(2)
	for (int i = 0; i < blockM + 1; i++)
	{
		for (int j = 0; j < blockN + 1; j++)
		{
			int loc0 = i + coords[0] * blockM, loc1 = j + coords[1] * blockN;
			if (loc1 == 0 || loc1 == N || loc0 == 0 || loc0 == M)
			{
				continue;
			}
			else if (i == 0 || j == 0)
			{
				continue;
			}
			else
			{
				resTemp[loc0 * (N + 1) + loc1] = u[loc0 * (N + 1) + loc1] - v[loc0 * (N + 1) + loc1];
			}
		}
	}
	MPI_Allreduce(resTemp.data(), res.data(), (M + 1) * (N + 1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

vector<double> funVmultConst(const vector<double> &u, double c)
{
	vector<double> res((M + 1) * (N + 1), 0);
	// #pragma omp parallel for
	for (int i = 1; i < M; i++)
	{
		for (int j = 1; j < N; j++)
		{
			res[i * (N + 1) + j] = u[i * (N + 1) + j] * c;
		}
	}
	return res;
}

void funVmultConstMPI(vector<double> &res, int *coords, const vector<double> &u, double c)
{
	vector<double> resTemp((M + 1) * (N + 1), 0);
#pragma omp parallel for collapse(2)
	for (int i = 0; i < blockM + 1; i++)
	{
		for (int j = 0; j < blockN + 1; j++)
		{
			int loc0 = i + coords[0] * blockM, loc1 = j + coords[1] * blockN;
			if (loc1 == 0 || loc1 == N || loc0 == 0 || loc0 == M)
			{
				continue;
			}
			else if (i == 0 || j == 0)
			{
				continue;
			}
			else
			{
				resTemp[loc0 * (N + 1) + loc1] = u[loc0 * (N + 1) + loc1] * c;
			}
		}
	}
	MPI_Allreduce(resTemp.data(), res.data(), (M + 1) * (N + 1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

// prepration for calculation
//
// define which area the 1/2node belongs to: 1 if inside D; remain 0 if outside D

void nodeTypeDef(vector<int> &nodeType)
{
	// #pragma omp parallel for
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (F(A_1 + (j + 0.5) * h_1, A_2 + (i + 0.5) * h_2) > 0)
			{
				nodeType[i * N + j] = 1;
				// cout << 1;
			}
		}
	}
}

void nodeTypeDefMPI(int rank, const int *coords, vector<int> &nodeType)
{
#pragma omp parallel for collapse(2)
	for (int i = 0; i < blockM; i++)
	{
		for (int j = 0; j < blockN; j++)
		{
			if (F(A_1 + (blockN * coords[1] + j + 0.5) * h_1, A_2 + (blockM * coords[0] + i + 0.5) * h_2) > 0)
			{
				nodeType[(i + blockM * coords[0]) * N + j + blockN * coords[1]] = 1;
			}
		}
	}
}
// calculate the right side of equation Aw=B
void FijDef(vector<double> &Fij, const vector<int> &nodeType)
{
#pragma omp parallel for
	for (int i = 1; i < M; i++)
	{
		for (int j = 1; j < N; j++)
		{
			int elem = nodeType[i * N + j];
			double yU = A_2 + (i + 0.5) * h_2;
			double yD = A_2 + (i - 0.5) * h_2;
			double xR = A_1 + (j + 0.5) * h_1;
			double xL = A_1 + (j - 0.5) * h_1;
			// cout<<"("<<xL<<","<<yD<<")"<< endl;
			// outside D
			if (xR <= -3 || yU <= 0 || yD >= 4 || xL >= 3 ||
				4 * xL + 3 * yD - 12 >= 0 ||
				4 * xR - 3 * yD + 12 <= 0)
			{
				continue;
			}
			// at least part of it is inside D
			else if (xR > -3 && xL < -3)
			{
				/*	1 - 4	*/
				int left = INDEX(nodeType, i - 1, j);
				switch (elem * 2 + left)
				{
				case 2: // 1
					Fij[i * N + j] = 1 * (((-3. / 4 * yU + 3 + xR) +
										   (3 + xR)) *
										  (yU / 2) / (h_1 * h_2));
					break;
				case 0: // 2
					Fij[i * N + j] = 1 * (3 + xR) * (4. / 3 * xR + 4) / 2 /
									 (h_1 * h_2);
					break;
				case 3: // 3
					Fij[i * N + j] = 1 * (((-3. / 4 * yU + 3 + xR) + (-3. / 4 * yD + 3 + xR)) * (h_2 / 2)) /
									 (h_1 * h_2);
					break;
				case 1: // 4
					Fij[i * N + j] = 1 * (-3. / 4 * yD + 3 + xR) *
									 (4. / 3 * xR + 4 - yD) / 2 /
									 (h_1 * h_2);
					break;
				}
			}
			else if (xR > 3 && xL < 3)
			{
				/*	5 - 8	*/
				if (INDEX(nodeType, i, j - 1) &&
					!INDEX(nodeType, i - 1, j - 1))
				{
					// 5
					Fij[i * N + j] = 1 * ((3. / 4 * yU - 3 - xL) + (3 - xL)) * yU / 2 / (h_1 * h_2);
				}
				else if (!INDEX(nodeType, i, j - 1) &&
						 !INDEX(nodeType, i - 1, j - 1))
				{
					// 6
					Fij[i * N + j] = 1 * ((3 - xL) * (-4. / 3 * xL + 4) / 2) /
									 (h_1 * h_2);
				}
				else if (INDEX(nodeType, i, j - 1) &&
						 INDEX(nodeType, i - 1, j - 1))
				{
					// 7
					Fij[i * N + j] = 1 * (((-3. / 4 * yU + 3 - xL) + (-3. / 4 * yD + 3 - xL)) * h_2 / 2) /
									 (h_1 * h_2);
				}
				else if (!INDEX(nodeType, i, j - 1) &&
						 INDEX(nodeType, i - 1, j - 1))
				{
					// 8
					Fij[i * N + j] = 1 * (((-4. / 3 * xL + 4 - yD) * (-3. / 4 * yD + 3 - xL)) / 2) /
									 (h_1 * h_2);
				}
			}
			else if (xL < 0 && xR > 0)
			{
				/*	9 - 36	*/
				if (yD < 0)
				{
					// 9
					if (elem && INDEX(nodeType, i, j - 1))
						Fij[i * N + j] = yU / h_2;
				}
				else if (yD > 0)
				{
					// 10
					if (elem && INDEX(nodeType, i, j - 1))
						Fij[i * N + j] = 1;
				}
				else if (yD > 0 && yU < 4)
				{
					// 11
					if (!elem && INDEX(nodeType, i - 1, j) &&
						INDEX(nodeType, i - 1, j - 1) &&
						!INDEX(nodeType, i, j - 1))
					{
						Fij[i * N + j] = 1 * (h_1 * h_2 - ((xR + 3. / 4 * yU - 3) * (yU + 4. / 3 * xR - 4) / 2) - (-xL + 3. / 4 * yU - 3) * (yU - 4. / 3 * xL + 4) / 2) /
										 (h_1 * h_2);
					}
				}
				else if (yD > 0 && yU > 4) // 12
				{
					if (!elem && INDEX(nodeType, i - 1, j) &&
						INDEX(nodeType, i - 1, j - 1) &&
						!INDEX(nodeType, i, j - 1))
					{
						Fij[i * N + j] = 1 * ((((-4. / 3 * xR + 4 - yD) + (4 - yD)) * xR / 2) + ((4. / 3 * xL + (4 - yD)) * (-xL / 2)) / (h_1 * h_2));
					}
				}
				else if (yU < 4) // 13-27
				{
					// 13-16
					if (yD > 0 && 4 * xL - 3 * yU + 12 < 0 && 4 * xL - 3 * yD + 12 < 0) // 13
					{
						Fij[i * N + j] = 1 * (((-3. / 4 * yD + 3 + (-3. / 4 * yU + 3)) * h_2) / (h_1 * h_2));
					}
					else if (yD > 0 && 4 * xL - 3 * yU + 12 < 0 && 4 * xL - 3 * yD + 12 > 0) // 14
					{
						Fij[i * N + j] = 1 * (((-3. / 4 * yD + 3 + (-3. / 4 * yU + 3)) * h_2 / 2) + ((-xL * h_2) - ((-xL + 3. / 4 * yU - 3) * (yU - 4. / 3 * xL - 4) / 2)) / (h_1 * h_2));
					}
					else if (yD < 0 && 4 * xL - 3 * yU + 12 < 0 && xL < -3) // 15
					{
						Fij[i * N + j] = 1 * ((3 + (-3. / 4 * yU + 3)) * yU) / (h_1 * h_2);
					}
					else if (yD < 0 && 4 * xL - 3 * yU + 12 < 0 && xL > -3) // 16
					{
						Fij[i * N + j] = 1 * ((3 + (-3. / 4 * yU + 3) * yU / 2) + (-xL * yU) - ((-xL + 3. / 4 * yU - 3) * (yU - 4. / 3 * xL - 4) / 2) / (h_1 * h_2));
					}
					// 17-18
					else if (4 * xL - 3 * yD + 12 < 0 && xR < 3 && 4 * xR + 3 * yU - 12 > 0)
					{
						if (yD > 0) // 17
						{
							Fij[i * N + j] = 1 * (((-3. / 4 * yD + 3) + (-3. / 4 * yU + 3)) * h_2 / 2 + (xR * h_2) - (yU + 4. / 3 * xR - 4) * (xR + 3. / 4 * yU - 3) / 2) / (h_1 * h_2);
						}
						else if (yD < 0) // 18
						{
							Fij[i * N + j] = 1 * ((-xL + 3 - 3. / 4 * yU) * yU - (3 - xR * (-4. / 3 * xR + 4)) / 2) / (h_1 * h_2);
						}
					}
					// 19
					else if (yD < 0 && xL > -3 && xR < 3 && 4 * xL - 3 * yU + 12 < 0 && 4 * xR + 3 * yU - 12 > 0)
					{
						Fij[i * N + j] = (h_1 * yU - (xR + 3. / 4 * yU - 3) * (yU + 4. / 3 * xR - 4) / 2 - (-xL + 3. / 4 * xL - 3) * (yU - 3. / 4 * xL + 4) / 2) / (h_1 * h_2);
					}
					else if (4 * xR + 3 * yU - 12 < 0 && 4 * xL - 3 * yU + 12 < 0) // 20-23
					{
						if (yD < 0 && xL > -3) // 20
						{
							Fij[i * N + j] = (xR * yU + (3 + 3 - 3. / 4 * yU) * yU / 2) / h_1 * h_2;
						}
						else if (yD < 0 && xL > -3) // 21
						{
							Fij[i * N + j] = (yU * h_1 - (yU - 4. / 3 * xL - 4) * (-xL + 3. / 4 * yU - 3) / 2) / (h_1 * h_2);
						}
						else if (yD > 0 && 4 * xL - 3 * yD + 12 < 0) // 22
						{
							Fij[i * N + j] = 1 * ((xR * h_2 + ((-3. / 4 * yU + 3) + (-3. / 4 * yD + 3)) * h_2 / 2) / (h_1 * h_2));
						}
						else if (yD > 0 && 4 * xL - 3 * yD + 12 > 0) // 23
						{
							Fij[i * N + j] = (h_1 * h_2 - (yU - 4. / 3 * xL - 4) * (-xL + 3. / 4 * yU - 3) / 2) / (h_1 * h_2);
						}
					}
					else if (4 * xL - 3 * yU + 12 > 0) // 24-27
					{
						if (yD < 0 && xR > 3) // 24
						{
							Fij[i * N + j] = ((-xL * yU) + (3 - 3. / 4 * yU + 3) * yU / 2) / (h_1 * h_2);
						}
						else if (yD < 0 && xR > 3) // 25
						{
							Fij[i * N + j] = (h_1 * yU - (yU + 4. / 3 * xR - 4) * (xR + 3. / 4 * yU - 3) / 2) / (h_1 * h_2);
						}
						else if (yD > 0 && 4 * xR + 3 * yD - 12 > 0) // 26
						{
							Fij[i * N + j] = (-xL * h_2 + (-3. / 4 * yD + 3 + (-3. / 4 * yU + 3)) * (h_2 / 2)) / (h_2 * h_1);
						}
						else if (yD > 0 && 4 * xR + 3 * yD - 12 < 0) // 27
						{
							Fij[i * N + j] = (h_1 * h_2 - (yU + 4. / 3 * xR - 4) * (xR + 3. / 4 * yU - 3) / 2) / (h_1 * h_2);
						}
					}
				}
				else if (yU > 4) // 28-35
				{
					if (yD > 0) // 28-31
					{
						if (4 * xR + 3 * yD - 12 > 0 && 4 * xL - 3 * yD + 12 < 0) // 28
						{
							Fij[i * N + j] = (4 - yD) * (-3. / 4 * yD + 3) / (h_1 * h_2);
						}
						else if (4 * xR + 3 * yD - 12 > 0 && 4 * xL - 3 * yD + 12 > 0) // 29
						{
							Fij[i * N + j] = (4 - yD) * (-3. / 4 * yD + 3) / 2 + ((4. / 3 * xL + 4 - yD) + (4 - yD)) * (-xL / 2) / (h_1 * h_2);
						}
						else if (4 * xR + 3 * yD - 12 < 0 && 4 * xL - 3 * yD + 12 < 0) // 30
						{
							Fij[i * N + j] = (4 - yD) * (-3. / 4 * yD + 3) / 2 + (-4. / 3 * xR + 4 - yD) + (4 - yD) * (xR / 2) / (h_1 * h_2);
						}
						else if (4 * xR + 3 * yD - 12 < 0 && 4 * xL - 3 * yD + 12 > 0) // 31
						{
							Fij[i * N + j] = ((-4. / 3 * xR + 4 - yD) + (4 - yD) * (xR / 2) + ((4. / 3 * xL + 4 - yD) + (4 - yD)) * (-xL / 2)) / (h_1 * h_2);
						}
					}
					else if (yD < 0) // 32-35
					{
						if (4 * xR + 3 * yD - 12 > 0 && xL < -3) // 32
						{
							Fij[i * N + j] = 12 / (h_1 * h_2);
						}
						else if (4 * xR + 3 * yD - 12 > 0 && xL > -3) // 33
						{
							Fij[i * N + j] = (12 - (3 + xL) * (4. / 3 * xL + 4) / 2) / (h_1 * h_2);
						}
						else if (4 * xL - 3 * yD + 12 < 0 && xR > 3) // 34
						{
							Fij[i * N + j] = (12 - (3 - xR) * (-4. / 3 * xR + 4) / 2) / (h_1 * h_2);
						}
						else if (xR < 3 && xL > -3) // 35
						{
							Fij[i * N + j] = (12 - (3 - xR) * (-4. / 3 * xR + 4) / 2 - (3 + xL) * (4. / 3 * xL + 4) / 2) / (h_1 * h_2);
						}
					}
				}
			}
		}
	}
}

void FijDefMPI(vector<double> &Fij, const vector<int> &nodeType)
{
#pragma omp parallel for
	for (int i = 1; i < M; i++)
	{
		for (int j = 1; j < N; j++)
		{
			int elem = nodeType[i * N + j];
			double yU = A_2 + (i + 0.5) * h_2;
			double yD = A_2 + (i - 0.5) * h_2;
			double xR = A_1 + (j + 0.5) * h_1;
			double xL = A_1 + (j - 0.5) * h_1;
			// cout<<"("<<xL<<","<<yD<<")"<< endl;
			// outside D
			if (xR <= -3 || yU <= 0 || yD >= 4 || xL >= 3 ||
				4 * xL + 3 * yD - 12 >= 0 ||
				4 * xR - 3 * yD + 12 <= 0)
			{
				continue;
			}
			else if (xR > -3 && xL < -3)
			{
				/*	1 - 4	*/
				int left = INDEX(nodeType, i - 1, j);
				switch (elem * 2 + left)
				{
				case 2: // 1
					Fij[i * N + j] = 1 * (((-3. / 4 * yU + 3 + xR) +
										   (3 + xR)) *
										  (yU / 2) / (h_1 * h_2));
					break;
				case 0: // 2
					Fij[i * N + j] = 1 * (3 + xR) * (4. / 3 * xR + 4) / 2 /
									 (h_1 * h_2);
					break;
				case 3: // 3
					Fij[i * N + j] = 1 * (((-3. / 4 * yU + 3 + xR) + (-3. / 4 * yD + 3 + xR)) * (h_2 / 2)) /
									 (h_1 * h_2);
					break;
				case 1: // 4
					Fij[i * N + j] = 1 * (-3. / 4 * yD + 3 + xR) *
									 (4. / 3 * xR + 4 - yD) / 2 /
									 (h_1 * h_2);
					break;
				}
			}
			else if (xR > 3 && xL < 3)
			{
				/*	5 - 8	*/
				if (INDEX(nodeType, i, j - 1) &&
					!INDEX(nodeType, i - 1, j - 1))
				{
					// 5
					Fij[i * N + j] = 1 * ((3. / 4 * yU - 3 - xL) + (3 - xL)) * yU / 2 / (h_1 * h_2);
				}
				else if (!INDEX(nodeType, i, j - 1) &&
						 !INDEX(nodeType, i - 1, j - 1))
				{
					// 6
					Fij[i * N + j] = 1 * ((3 - xL) * (-4. / 3 * xL + 4) / 2) /
									 (h_1 * h_2);
				}
				else if (INDEX(nodeType, i, j - 1) &&
						 INDEX(nodeType, i - 1, j - 1))
				{
					// 7
					Fij[i * N + j] = 1 * (((-3. / 4 * yU + 3 - xL) + (-3. / 4 * yD + 3 - xL)) * h_2 / 2) /
									 (h_1 * h_2);
				}
				else if (!INDEX(nodeType, i, j - 1) &&
						 INDEX(nodeType, i - 1, j - 1))
				{
					// 8
					Fij[i * N + j] = 1 * (((-4. / 3 * xL + 4 - yD) * (-3. / 4 * yD + 3 - xL)) / 2) /
									 (h_1 * h_2);
				}
			}
			else if (xL < 0 && xR > 0)
			{
				/*	9 - 36	*/
				if (yD < 0)
				{
					// 9
					if (elem && INDEX(nodeType, i, j - 1))
						Fij[i * N + j] = yU / h_2;
				}
				else if (yD > 0)
				{
					// 10
					if (elem && INDEX(nodeType, i, j - 1))
						Fij[i * N + j] = 1;
				}
				else if (yD > 0 && yU < 4)
				{
					// 11
					if (!elem && INDEX(nodeType, i - 1, j) &&
						INDEX(nodeType, i - 1, j - 1) &&
						!INDEX(nodeType, i, j - 1))
					{
						Fij[i * N + j] = 1 * (h_1 * h_2 - ((xR + 3. / 4 * yU - 3) * (yU + 4. / 3 * xR - 4) / 2) - (-xL + 3. / 4 * yU - 3) * (yU - 4. / 3 * xL + 4) / 2) /
										 (h_1 * h_2);
					}
				}
				else if (yD > 0 && yU > 4) // 12
				{
					if (!elem && INDEX(nodeType, i - 1, j) &&
						INDEX(nodeType, i - 1, j - 1) &&
						!INDEX(nodeType, i, j - 1))
					{
						Fij[i * N + j] = 1 * ((((-4. / 3 * xR + 4 - yD) + (4 - yD)) * xR / 2) + ((4. / 3 * xL + (4 - yD)) * (-xL / 2)) / (h_1 * h_2));
					}
				}
				else if (yU < 4) // 13-27
				{
					// 13-16
					if (yD > 0 && 4 * xL - 3 * yU + 12 < 0 && 4 * xL - 3 * yD + 12 < 0) // 13
					{
						Fij[i * N + j] = 1 * (((-3. / 4 * yD + 3 + (-3. / 4 * yU + 3)) * h_2) / (h_1 * h_2));
					}
					else if (yD > 0 && 4 * xL - 3 * yU + 12 < 0 && 4 * xL - 3 * yD + 12 > 0) // 14
					{
						Fij[i * N + j] = 1 * (((-3. / 4 * yD + 3 + (-3. / 4 * yU + 3)) * h_2 / 2) + ((-xL * h_2) - ((-xL + 3. / 4 * yU - 3) * (yU - 4. / 3 * xL - 4) / 2)) / (h_1 * h_2));
					}
					else if (yD < 0 && 4 * xL - 3 * yU + 12 < 0 && xL < -3) // 15
					{
						Fij[i * N + j] = 1 * ((3 + (-3. / 4 * yU + 3)) * yU) / (h_1 * h_2);
					}
					else if (yD < 0 && 4 * xL - 3 * yU + 12 < 0 && xL > -3) // 16
					{
						Fij[i * N + j] = 1 * ((3 + (-3. / 4 * yU + 3) * yU / 2) + (-xL * yU) - ((-xL + 3. / 4 * yU - 3) * (yU - 4. / 3 * xL - 4) / 2) / (h_1 * h_2));
					}
					// 17-18
					else if (4 * xL - 3 * yD + 12 < 0 && xR < 3 && 4 * xR + 3 * yU - 12 > 0)
					{
						if (yD > 0) // 17
						{
							Fij[i * N + j] = 1 * (((-3. / 4 * yD + 3) + (-3. / 4 * yU + 3)) * h_2 / 2 + (xR * h_2) - (yU + 4. / 3 * xR - 4) * (xR + 3. / 4 * yU - 3) / 2) / (h_1 * h_2);
						}
						else if (yD < 0) // 18
						{
							Fij[i * N + j] = 1 * ((-xL + 3 - 3. / 4 * yU) * yU - (3 - xR * (-4. / 3 * xR + 4)) / 2) / (h_1 * h_2);
						}
					}
					// 19
					else if (yD < 0 && xL > -3 && xR < 3 && 4 * xL - 3 * yU + 12 < 0 && 4 * xR + 3 * yU - 12 > 0)
					{
						Fij[i * N + j] = (h_1 * yU - (xR + 3. / 4 * yU - 3) * (yU + 4. / 3 * xR - 4) / 2 - (-xL + 3. / 4 * xL - 3) * (yU - 3. / 4 * xL + 4) / 2) / (h_1 * h_2);
					}
					else if (4 * xR + 3 * yU - 12 < 0 && 4 * xL - 3 * yU + 12 < 0) // 20-23
					{
						if (yD < 0 && xL > -3) // 20
						{
							Fij[i * N + j] = (xR * yU + (3 + 3 - 3. / 4 * yU) * yU / 2) / h_1 * h_2;
						}
						else if (yD < 0 && xL > -3) // 21
						{
							Fij[i * N + j] = (yU * h_1 - (yU - 4. / 3 * xL - 4) * (-xL + 3. / 4 * yU - 3) / 2) / (h_1 * h_2);
						}
						else if (yD > 0 && 4 * xL - 3 * yD + 12 < 0) // 22
						{
							Fij[i * N + j] = 1 * ((xR * h_2 + ((-3. / 4 * yU + 3) + (-3. / 4 * yD + 3)) * h_2 / 2) / (h_1 * h_2));
						}
						else if (yD > 0 && 4 * xL - 3 * yD + 12 > 0) // 23
						{
							Fij[i * N + j] = (h_1 * h_2 - (yU - 4. / 3 * xL - 4) * (-xL + 3. / 4 * yU - 3) / 2) / (h_1 * h_2);
						}
					}
					else if (4 * xL - 3 * yU + 12 > 0) // 24-27
					{
						if (yD < 0 && xR > 3) // 24
						{
							Fij[i * N + j] = ((-xL * yU) + (3 - 3. / 4 * yU + 3) * yU / 2) / (h_1 * h_2);
						}
						else if (yD < 0 && xR > 3) // 25
						{
							Fij[i * N + j] = (h_1 * yU - (yU + 4. / 3 * xR - 4) * (xR + 3. / 4 * yU - 3) / 2) / (h_1 * h_2);
						}
						else if (yD > 0 && 4 * xR + 3 * yD - 12 > 0) // 26
						{
							Fij[i * N + j] = (-xL * h_2 + (-3. / 4 * yD + 3 + (-3. / 4 * yU + 3)) * (h_2 / 2)) / (h_2 * h_1);
						}
						else if (yD > 0 && 4 * xR + 3 * yD - 12 < 0) // 27
						{
							Fij[i * N + j] = (h_1 * h_2 - (yU + 4. / 3 * xR - 4) * (xR + 3. / 4 * yU - 3) / 2) / (h_1 * h_2);
						}
					}
				}
				else if (yU > 4) // 28-35
				{
					if (yD > 0) // 28-31
					{
						if (4 * xR + 3 * yD - 12 > 0 && 4 * xL - 3 * yD + 12 < 0) // 28
						{
							Fij[i * N + j] = (4 - yD) * (-3. / 4 * yD + 3) / (h_1 * h_2);
						}
						else if (4 * xR + 3 * yD - 12 > 0 && 4 * xL - 3 * yD + 12 > 0) // 29
						{
							Fij[i * N + j] = (4 - yD) * (-3. / 4 * yD + 3) / 2 + ((4. / 3 * xL + 4 - yD) + (4 - yD)) * (-xL / 2) / (h_1 * h_2);
						}
						else if (4 * xR + 3 * yD - 12 < 0 && 4 * xL - 3 * yD + 12 < 0) // 30
						{
							Fij[i * N + j] = (4 - yD) * (-3. / 4 * yD + 3) / 2 + (-4. / 3 * xR + 4 - yD) + (4 - yD) * (xR / 2) / (h_1 * h_2);
						}
						else if (4 * xR + 3 * yD - 12 < 0 && 4 * xL - 3 * yD + 12 > 0) // 31
						{
							Fij[i * N + j] = ((-4. / 3 * xR + 4 - yD) + (4 - yD) * (xR / 2) + ((4. / 3 * xL + 4 - yD) + (4 - yD)) * (-xL / 2)) / (h_1 * h_2);
						}
					}
					else if (yD < 0) // 32-35
					{
						if (4 * xR + 3 * yD - 12 > 0 && xL < -3) // 32
						{
							Fij[i * N + j] = 12 / (h_1 * h_2);
						}
						else if (4 * xR + 3 * yD - 12 > 0 && xL > -3) // 33
						{
							Fij[i * N + j] = (12 - (3 + xL) * (4. / 3 * xL + 4) / 2) / (h_1 * h_2);
						}
						else if (4 * xL - 3 * yD + 12 < 0 && xR > 3) // 34
						{
							Fij[i * N + j] = (12 - (3 - xR) * (-4. / 3 * xR + 4) / 2) / (h_1 * h_2);
						}
						else if (xR < 3 && xL > -3) // 35
						{
							Fij[i * N + j] = (12 - (3 - xR) * (-4. / 3 * xR + 4) / 2 - (3 + xL) * (4. / 3 * xL + 4) / 2) / (h_1 * h_2);
						}
					}
				}
			}
		}
	}
}

// coefficients, determined by integral and will be calculated during the difference
// coefficient A
double coefA(int i, int j, const vector<int> &nodeType)
{
	double x = A_1 + (j - 0.5) * h_1;
	double yU = A_2 + (i + 0.5) * h_2;
	double yD = A_2 + (i - 0.5) * h_2;
	int U = nodeType[i * N + j - 1];
	int D = nodeType[(i - 1) * N + j - 1];
	// if the two endpoints of the segment are outside d
	// if (nodeType[i - 1][j - 1] + nodeType[i][j - 1] == 0) {
	if (!(D + U))
	{
		if (x < -3 || x > 3 || yU < 0 || yD > 4 || 4 * x + 3 * yD - 12 > 0 || 4 * x - 3 * yD + 12 < 0)
		{
			return 1 / eps;
		}
		else if (x > 0 && x < 3)
		{
			return (-4. / 3 * x + 4) / h_2 + (1 - (-4. / 3 * x + 4) / h_2) / eps;
		}
		else if (x > -3 && x < 0)
		{
			return (4. / 3 * x + 4) / h_2 + (1 - (4. / 3 * x + 4) / h_2) / eps;
		}
	}
	else if (D + U == 1)
	{
		if (yD < 0)
		{
			return (yU) / h_2 + (1 - yU / h_2) / eps;
		}
		else
		{
			if (x > 0 && x < 3)
			{
				return (-4. / 3 * x + 4 - yD) / h_2 + (1 - (-4. / 3 * x + 4 - yD) / h_2) / eps;
			}
			else if (x > -3 && x < 0)
			{
				return (4. / 3 * x + 4 - yD) / h_2 + (1 - (4. / 3 * x + 4 - yD)) / h_2 / eps;
			}
		}
	}
	else
	{
		return 1;
	}
}

// coefficient B
double coefB(int i, int j, const vector<int> &nodeType)
{

	double y = A_2 + (i - 0.5) * h_2;
	double xR = A_1 + (j + 0.5) * h_1;
	double xL = A_1 + (j - 0.5) * h_1;
	int R = nodeType[(i - 1) * N + j];
	int L = nodeType[(i - 1) * N + j - 1];
	// if two endpoints of the segment are outside d
	// if (nodeType[i - 1][j - 1] + nodeType[i - 1][j] == 0) {
	if (L + R == 0)
	{
		if (y < 0 || y > 4 || xL > 3 || xR < -3 || 4 * xL + 3 * y - 12 > 0 || 4 * xR - 3 * y + 12 < 0)
		{
			return 1 / eps;
		}
		else
		{
			return (-3. / 2 * y + 6) / h_1 + (1 - (-3. / 2 * y + 6) / h_1) / eps;
		}
	}
	// if one of the two endpoints of the segment is outside d
	// else if (nodeType[i - 1][j - 1] + nodeType[i][j] == 1) {
	else if (L + R == 1)
	{
		if (xL < 3 && xL > -3)
		{
			return (-3. / 4 * y + 3 - xL) / h_1 + (1 - (-3. / 4 * y + 3 - xL) / h_1) / eps;
		}
		else if (xR > -3 && xR < 3)
		{
			return (-3. / 4 * y + 3 + xR) / h_1 + (1 - (-3. / 4 * y + 3 + xR) / h_1) / eps;
		}
	}
	else
	{
		return 1;
	}
}

// perform A(w) and return the result, My w is defined in the entire rectangular space so that the subscripts of the internal points are the same as in the document.
auto operatorA(const vector<double> &w, const vector<int> &nodeType)
{
	vector<double> res((M + 1) * (N + 1), 0);
	// #pragma omp parallel for
	for (int i = 1; i < M; i++)
	{
		for (int j = 1; j < N; j++)
		{
			res[i * (N + 1) + j] = -(coefA(i, j + 1, nodeType) * (w[i * (N + 1) + j + 1] - w[i * (N + 1) + j]) / h_1 - coefA(i, j, nodeType) * (w[i * (N + 1) + j] - w[i * (N + 1) + j - 1]) / h_1) / (h_1) - (coefB(i + 1, j, nodeType) * (w[(i + 1) * (N + 1) + j] - w[i * (N + 1) + j]) / h_2 - coefB(i, j, nodeType) * (w[i * (N + 1) + j] - w[(i - 1) * N + j]) / h_2) / (h_2);
		}
	}
	return res;
}

void operatorAMPI(int rank, vector<double> &res, const int *coords, const vector<double> &w, const vector<int> &nodeType)
{
	vector<double> resTemp((M + 1) * (N + 1), 0);
#pragma omp parallel for collapse(2)
	for (int i = 0; i < blockM + 1; i++)
	{
		for (int j = 0; j < blockN + 1; j++)
		{
			int loc0 = i + coords[0] * blockM, loc1 = j + coords[1] * blockN;
			int realCoords[2]{i + coords[0] * blockM, j + coords[1] * blockN};
			/// boundary points won't be calculated, they don't participate in calculation and always equal 0
			if (loc1 == 0 || loc1 == N || loc0 == 0 || loc0 == M)
			{
				continue;
			}
			// those points that are included in the calculation
			else if (i == 0 || j == 0)
			{
				continue;
			}
			else
			{
				resTemp[loc0 * (N + 1) + loc1] = -(coefA(loc0, loc1 + 1, nodeType) * (w[loc0 * (N + 1) + loc1 + 1] - w[loc0 * (N + 1) + loc1]) / h_1 - coefA(loc0, loc1, nodeType) * (w[loc0 * (N + 1) + loc1] - w[loc0 * (N + 1) + loc1 - 1]) / h_1) / (h_1) - (coefB(loc0 + 1, loc1, nodeType) * (w[(loc0 + 1) * (N + 1) + loc1] - w[loc0 * (N + 1) + loc1]) / h_2 - coefB(loc0, loc1, nodeType) * (w[loc0 * (N + 1) + loc1] - w[(loc0 - 1) * (N + 1) + loc1]) / h_2) / (h_2);

				////internal
				// if (i != 0 || i != blockM || j != 0 || j != blockN) {
				//	res[loc0 * N + loc1] = -(coefA(loc0, loc1 + 1, nodeType) * (w[loc0 * N + loc1 + 1] - w[loc0 * N + loc1]) / h_1 - coefA(loc0, loc1, nodeType) * (w[loc0 * N + loc1] - w[loc0 * N + loc1 - 1]) / h_1) / (h_1) - (coefB(loc0 + 1, loc1, nodeType) * (w[(loc0 + 1) * N + loc1] - w[loc0 * N + loc1]) / h_2 - coefB(loc0, loc1, nodeType) * (w[loc0 * N + loc1] - w[(loc0 - 1) * N + loc1]) / h_2) / (h_2);
				// }
				////boundary, data exchanges occur
				// else {
				//	int maxC0, maxC1;
				//	MPI_Allreduce(&coords[1], &maxC1, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
				//	MPI_Allreduce(&coords[0], &maxC0, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
				//	int leftRank = rank - 1, rightRank = rank + 1;
				//	int upRank = rank + maxC1, downRank = rank - maxC1;
				//	//vector<double> sendTop(blockN,0);
				//	//vector<double> sendBottom(blockN,0);
				//	vector<double> sendLeft(blockM,0);
				//	vector<double> sendRight(blockM,0);
				//	vector<double> recvTop(blockN,0);
				//	vector<double> recvBottom(blockN,0);
				//	vector<double> recvLeft(blockM,0);
				//	vector<double> recvRight(blockM,0);
				//	//cout << maxC1 << " " << maxC2;
				//	if (coords[0] == 0) {
				//		if (coords[1] == 0) {
				//			MPI_Sendrecv(&w[loc0 * N + loc1], blockM, MPI_DOUBLE, upRank, 0, recvTop.data(), blockM, MPI_DOUBLE, upRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//			for (int k = 0; k < blockM; k++) {
				//				sendRight[k] = w[(loc0 + k) * N + loc1];
				//			}
				//			MPI_Sendrecv(sendRight.data(), blockM, MPI_DOUBLE, rightRank, 0, recvRight.data(), blockM, MPI_DOUBLE, rightRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//			res[loc0 * N + loc1] = -(coefA(loc0, loc1 + 1, nodeType) * (recvRight[0] - w[loc0 * N + loc1]) / h_1 - coefA(loc0, loc1, nodeType) * (w[loc0 * N + loc1] - w[loc0 * N + loc1 - 1]) / h_1) / (h_1) - (coefB(loc0 + 1, loc1, nodeType) * (recvTop[0] - w[loc0 * N + loc1]) / h_2 - coefB(loc0, loc1, nodeType) * (w[loc0 * N + loc1] - w[(loc0 - 1) * N + loc1]) / h_2) / (h_2);
				//		}
				//		else if (coords[1] == maxC1) {
				//			MPI_Sendrecv(&w[loc0 * N + loc1], blockN, MPI_DOUBLE, rightRank, 0, recvRight.data(), blockN, MPI_DOUBLE, rightRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//			MPI_Sendrecv(&w[loc0 * N + loc1], blockM, MPI_DOUBLE, downRank, 0, recvBottom.data(), blockM, MPI_DOUBLE, downRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//			res[loc0 * N + loc1] = -(coefA(loc0, loc1 + 1, nodeType) * (recvRight[0] - w[loc0 * N + loc1]) / h_1 - coefA(loc0, loc1, nodeType) * (w[loc0 * N + loc1] - w[loc0 * N + loc1 - 1]) / h_1) / (h_1) - (coefB(loc0 + 1,))
				//		}
				//		else {
				//										MPI_Sendrecv(&w[loc0 * N + loc1], blockN, MPI_DOUBLE, rightRank, 0, recvRight.data(), blockN, MPI_DOUBLE, rightRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//			res[loc0 * N + loc1] = -(coefA(loc0, loc1 + 1, nodeType) * (recvRight[0] - w[loc0 * N + loc1]) / h_1 - coefA(loc0, loc1, nodeType) * (w[loc0 * N + loc1] - w[loc0 * N + loc1 - 1]) / h_1) / (h_1) - (coefB(loc0 + 1, loc1, nodeType) * (w[(loc0 + 1) * N + loc1] - w[loc0 * N + loc1]) / h_2 - coefB(loc0, loc1, nodeType) * (w[loc0 * N + loc1] - w[(loc0 - 1) * N + loc1]) / h_2) / (h_2);0
				//		}
				//	}
				//	else if (coords[0] == maxC0) {

				//		}
				//	else if (coords[1] == 0) {

				//	}
				//	else if (coords[1] == maxC1) {

				//	}
				//	else {

				//	}
				//}
			}
			// nodeType[(i + blockM * coords[1]) * N + j + blockN * coords[0]] = 1;
			// res[(i + blockM * coords[1]) * N + j + blockN * coords[0]] = -(coefA((i + blockM * coords[1]) * N, j + blockN * coords[0] + 1, nodeType) * (w[(i + blockM * coords[1]) * N + j + blockN * coords[0] + 1] - w[(i + blockM * coords[1]) * N + j + blockN * coords[0]]) / h_1 - coefA((i + blockM * coords[1]) * N, j + blockN * coords[0], nodeType) * (w[(i + blockM * coords[1]) * N + j + blockN * coords[0]] - w[(i + blockM * coords[1]) * N + j + blockN * coords[0] - 1]) / h_1) / (h_1)-(coefB((i + 1 + blockM * coords[1]) * N, j + blockN * coords[0], nodeType) * (w[(i + 1 + blockM * coords[1]) * N + j + blockN * coords[0]] - w[(i + blockM * coords[1]) * N + j + blockN * coords[0]]) / h_2 - coefB((i + blockM * coords[1]) * N, j + blockN * coords[0], nodeType) * (w[(i + blockM * coords[1]) * N + j + blockN * coords[0]] - w[(i - 1 + blockM * coords[1]) * N + j + blockN * coords[0]]) / h_2) / (h_2);
		}
	}
	MPI_Allreduce(resTemp.data(), res.data(), (M + 1) * (N + 1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
	int provided;
	// initialize MPI
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided != MPI_THREAD_MULTIPLE)
	{
		cout << "MPI do not Support Multiple thread" << endl;
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
	int rank, numProcs;

	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int dims[2]{0, 0};
	MPI_Dims_create(numProcs, 2, dims);
	int periods[2]{0, 0};
	MPI_Comm MPI_COMM_CART;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &MPI_COMM_CART);

	// Get my coordinates in the new communicator
	int cartCoords[2];
	MPI_Cart_coords(MPI_COMM_CART, rank, 2, cartCoords);

	// Print my location in the 2D torus.
	// printf("[MPI process %d] I am located at (%d, %d).\n", rank, cartCoords[0], cartCoords[1]);

	domainSplit(blockM, blockN, numProcs);
	auto nodeType = vector<int>(M * N, 0);
	nodeTypeDefMPI(rank, cartCoords, nodeType);
	vector<int> nodeTypeReduce(M * N, 0);
	MPI_Allreduce(nodeType.data(), nodeTypeReduce.data(), M * N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	auto Fij = vector<double>((M + 1) * (N + 1), 0);
	FijDef(Fij, nodeTypeReduce);
	int iterNum = 0;
	vector<double> wNewer((M + 1) * (N + 1), 0);
	double Arr, ArAr;
	vector<double> residual((M + 1) * (N + 1), 0);
	vector<double> Ar((M + 1) * (N + 1), 0);
	vector<double> Aw((M + 1) * (N + 1), 0);
	vector<double> tr((M + 1) * (N + 1), 0);
	vector<double> wErr((M + 1) * (N + 1), 0);
	double tau, error;
	// Generalized minimal residual method
	// auto start = std::chrono::steady_clock::now();
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();
	do
	{
		w = wNewer;
		fill(wNewer.begin(), wNewer.end(), 0);
		fill(Aw.begin(), Aw.end(), 0);

		operatorAMPI(rank, Aw, cartCoords, w, nodeTypeReduce);

		fill(residual.begin(), residual.end(), 0);
		funVSubtractMPI(residual, cartCoords, Aw, Fij);
		fill(Ar.begin(), Ar.end(), 0);
		operatorAMPI(rank, Ar, cartCoords, residual, nodeTypeReduce);
		// funVScalProd(Ar, residual, h_1, h_2) / pow(funVNorm(Ar, h_1, h_2), 2);
		funVScalProdMPI(Arr, Ar, residual, cartCoords, h_1, h_2);
		funVScalProdMPI(ArAr, Ar, Ar, cartCoords, h_1, h_2);
		tau = Arr / ArAr;
		fill(tr.begin(), tr.end(), 0);
		funVmultConstMPI(tr, cartCoords, residual, tau);
		// wNewer = funVSubtract(w, funVmultConst(residual, tau));
		funVSubtractMPI(wNewer, cartCoords, w, tr);
		iterNum++;
		fill(wErr.begin(), wErr.end(), 0);
		funVSubtractMPI(wErr, cartCoords, wNewer, w);
		funVNormMPI(error, wErr, cartCoords, h_1, h_2);
		// cout  << "error: " << error <<endl;
	} while (error >= delta);
	//} while (0);
	MPI_Barrier(MPI_COMM_WORLD);
	double end = MPI_Wtime();
	if (rank == 0)
	{
		cout << "Execution succeed!" << endl;
		cout << "Total iteration: " << iterNum << endl;
		cout << "Time elapsed: " << (end - start) << "s" << endl;
	}
	/// store result as .csv file and information about current execution as .txt file
#pragma omp parallel
	{
#pragma omp single
		{
			if (rank == 0)
			{
				string txtFileName("ParallelInfo"), csvFileName("MPI_OpenMP_Data");
				txtFileName += to_string(M) + " " + to_string(N) + " " + to_string(numProcs) + " " + to_string(omp_get_num_threads()) + ".txt";
				csvFileName += to_string(M) + " " + to_string(N) + " " + to_string(numProcs) + " " + to_string(omp_get_num_threads()) + ".csv";
				ofstream INFOFILE(txtFileName), CSVFILE(csvFileName);
				INFOFILE << "Total iteration: " << iterNum << endl;
				INFOFILE << "Time elapsed: " << (end - start) << "s" << endl;
				INFOFILE << "num_cores = " << numProcs << endl;
				INFOFILE << "num_threads = " << omp_get_num_threads() << endl;
				INFOFILE << "Solution:" << endl;
				// auto A = operatorA(operatorA(w, nodeType), nodeType);
				// print whole w
				for (int i = M; i >= 0; i--)
				{
					for (int j = 0; j < N + 1; j++)
					{
						INFOFILE << setw(8) << setprecision(4) << wNewer[i * (N + 1) + j] << " ";
						if (j <= N - 1)
						{
							CSVFILE << wNewer[(M - i) * (N + 1) + j] << ",";
						}
						else
						{
							CSVFILE << wNewer[(M - i) * (N + 1) + j] << "\n";
						}
					}
					INFOFILE << endl;
				}
				INFOFILE.close();
				CSVFILE.close();
			}
		}
	}
	MPI_Finalize();
	return 0;
}