#define BLOCK_SIZE 38000
#define V_SIZE 200000

__kernel
__attribute((reqd_work_group_size(1,1,1)))
void cg(
	 __global float *restrict X_result,
	 __global const float *restrict VAL,
	 __global const int *restrict COL_IND,
	 __global const int *restrict ROW_PTR,
	 __global const float *restrict B,
	 const int N,
	 const int K,
	 const int VAL_SIZE
	 )
{
	float x[BLOCK_SIZE], r[BLOCK_SIZE], p[BLOCK_SIZE], y[BLOCK_SIZE], alfa, beta;
	float VAL_local[V_SIZE];
	int COL_IND_local[V_SIZE], ROW_PTR_local[BLOCK_SIZE + 1];
	float temp_sum=0.0f, temp_pap, temp_rr1, temp_rr2;

	temp_rr1 = 0.0f;

	for(int i = 0; i < N; ++i){
		ROW_PTR_local[i] = ROW_PTR[i];
		x[i] = 0.0f;
		r[i] = B[i];
		p[i] = B[i];
		temp_rr1 += r[i] * r[i];
	}
	ROW_PTR_local[N] = ROW_PTR[N];


	for(int i = 0; i < VAL_SIZE; ++i){
		COL_IND_local[i] = COL_IND[i];
		VAL_local[i] = VAL[i];
	}


	for(int i = 0; i < K; ++i){
		temp_pap = 0.0f;
		int j = 0, l = ROW_PTR_local[0];

		for(int m = 0; m < VAL_SIZE-ROW_PTR_local[0]; ++m){
			temp_sum += p[COL_IND_local[l]] * VAL_local[l];
			++l;
			if(l == ROW_PTR_local[j + 1]) {
				y[j] = temp_sum;
				temp_pap += p[j] * temp_sum;
				temp_sum = 0.0f;
				++j;
				l = ROW_PTR_local[j];
			}
		}

		alfa = temp_rr1 / temp_pap;

		temp_rr2 = 0.0f;

		for(int j = 0; j < N; ++j){
			x[j] += alfa * p[j];
			r[j] -= alfa * y[j];
			temp_rr2 += r[j] * r[j];
		}

		beta = temp_rr2 / temp_rr1;


		for(int j = 0; j < N; ++j){
			p[j] = r[j] + beta * p[j];
		}
		temp_rr1 = temp_rr2;

	}

	for(int j = 0; j < N; ++j){
		X_result[j] = x[j];
	}
}