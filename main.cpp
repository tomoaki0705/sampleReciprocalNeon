#include <iostream>
#include <arm_neon.h>
#ifndef __ARM_NEON__
#error // activate neon
#endif

void compareArray(int *normal, int* neon, int a, int b, int iteration)
{
	// compare each element of vector
	for(int i = 0;i	< 4;i++)
	{
		if(normal[i] != neon[i])
		{
			// show detail when compute result doesn't match
			std::cout << "iteration:" << iteration;
			std::cout << " expected:" << normal[i];
			std::cout << " actual:" << neon[i];
			std::cout << " " << a+i << "/" << b+i << std::endl;
		}
	}
}

void doNewtonRaphson(float* bufA, float* bufB, int* resultNormal, int a, int b, int iteration)
{
	// compute A/B
	float32x4_t vectorA = vld1q_f32(bufA);
	float32x4_t vectorB = vld1q_f32(bufB);
	
	// initial estimation
	float32x4_t reciprocal = vrecpeq_f32(vectorB);
	float32x4_t result = vmulq_f32(vectorA, reciprocal);

	int resultNEON[] = {0, 0, 0, 0, };
	if(iteration == 0)
	{
		// Just using reciprocal
		vst1q_s32(resultNEON, vcvtq_s32_f32(result));
		compareArray(resultNormal, resultNEON, a, b, 0);
	}
	else
	{
		// do Newton Raphson estimation
		for(int i = 0;i < iteration;i++)
		{
			reciprocal = vmulq_f32(vrecpsq_f32(vectorB, reciprocal), reciprocal);
			result = vmulq_f32(vectorA, reciprocal);
		}
		if(iteration == 3)
		{
			float32x4_t _05 = vdupq_n_f32(0.5f);
			result = vaddq_f32(result, _05);
		}
		vst1q_s32(resultNEON, vcvtq_s32_f32(result));
		compareArray(resultNormal, resultNEON, a, b, iteration);
	}
}

void testReciprocal(int a, int b)
{
	int resultNormal[] = {0, 0, 0, 0, };
	float bufA[4], bufB[4];
	// compute (a/b), (a+1)/(b+1), (a+2)/(b+2), (a+3)/(b+3)
	for(int i = 0;i < 4;i++)
	{
		float fa = (float)(a+i);
		float fb = (float)(b+i);
		resultNormal[i] = (int)(fa / fb);
		bufA[i] = (float)(a+i);
		bufB[i] = (float)(b+i);
	}

	doNewtonRaphson(bufA, bufB, resultNormal, a, b, 0);
	doNewtonRaphson(bufA, bufB, resultNormal, a, b, 1);
	doNewtonRaphson(bufA, bufB, resultNormal, a, b, 2);
}

int main(int argc, char** argv)
{
	for(int i = -128;i <= 128-4;i+=4)
	{
	for(int j = 0;j < 255;j++)
	{
		testReciprocal(i, i+j);
	}
	}
	return 0;
}
