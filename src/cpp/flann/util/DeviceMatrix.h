#pragma once

#include <flann/util/matrix.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>


namespace flann
{
	namespace cuda
	{
		struct step_functor : public thrust::unary_function<int, int>
		{
			int columns;
			int step;
			step_functor(int columns_, int step_) : columns(columns_), step(step_)  {   };
			step_functor() : columns(0), step(0){}
			__host__ __device__
				int operator()(int x) const
			{
				int row = x / columns;
				int idx = (row * step) + x % columns;
				return idx;
			}
		}; // step_functor

		struct Range
		{
			Range(int start_, int end_ = -1) : start(start_), end(end_){}
			int start;
			int end;
			__device__ __host__ int span()
			{
				return end - start;
			}
		}; // Range
		// TODO
		// Ideally we would have an object that can be initialized with some information about the matrix, the object then can be combined with an iterator to
		// Such that when the iterator is dereferenced it presents an object to SplitNodes2::operator() that can be used to access all row elements 
		template<typename T>struct Row
		{
			thrust::device_ptr<T> ptr;
			size_t step; // Step is in elements, not bytes
		}; // Row


		template<typename T>
		struct DeviceMatrix : public ::flann::Matrix<T>
		{
			typedef thrust::permutation_iterator<thrust::device_ptr<T>, thrust::transform_iterator<step_functor, thrust::counting_iterator<size_t>>> iterator;

			__device__ __host__ DeviceMatrix(T* data_, size_t rows_, size_t cols_, size_t stride_ = 0) :
				::flann::Matrix<T>(data_, rows_, cols_, stride_)
			{
                if (Matrix_::stride == 0)
                    Matrix_::stride = sizeof(T)*Matrix_::cols;
			}
			__device__ __host__ DeviceMatrix() : ::flann::Matrix<T>(){}

			__host__ DeviceMatrix(flann::Matrix<T>& other) : ::flann::Matrix<T>(other){}


			__host__ void operator=(flann::Matrix<T>& other)
			{
				flann::Matrix<T>::operator=(other);
			}
			__forceinline__ __device__ __host__  thrust::device_ptr<T> operator[](size_t rowIndex) const
			{
                return thrust::device_pointer_cast(reinterpret_cast<T*>(Matrix_::data + rowIndex*Matrix_::stride));
			}

			__forceinline__ __device__ __host__ thrust::device_ptr<T> ptr() const
			{
                return thrust::device_pointer_cast(reinterpret_cast<T*>(Matrix_::data));
			}
			__device__ __host__ iterator begin(int col = -1)
			{
				if (col == -1)
				{
					return thrust::make_permutation_iterator(
						ptr(),
						thrust::make_transform_iterator(
						thrust::make_counting_iterator(size_t(0)),
                        step_functor(Matrix_::cols, Matrix_::stride / sizeof(T))));
				}
				else
				{
					return thrust::make_permutation_iterator(
						ptr() + col,
						thrust::make_transform_iterator(
						thrust::make_counting_iterator(size_t(0)),
                        step_functor(1, Matrix_::stride / sizeof(T))));
				}
			}
			__device__ __host__ iterator end(int col = -1)
			{
				if (col == -1)
				{
                    size_t size = Matrix_::rows*Matrix_::cols;
					return thrust::make_permutation_iterator(
						ptr(),
						thrust::make_transform_iterator(
						thrust::make_counting_iterator(size),
                        step_functor(Matrix_::cols, Matrix_::stride / sizeof(T))));
				}
				else
				{

					return thrust::make_permutation_iterator(
						ptr() + col,
						thrust::make_transform_iterator(
                        thrust::make_counting_iterator(Matrix_::rows),
                        step_functor(1, Matrix_::stride / sizeof(T))));
				}
			}

			iterator begin(Range rowRange, Range colRange = Range(-1, -1))
			{
				if (rowRange.start == -1)
					rowRange.start = 0;
				if (rowRange.end == -1)
                    rowRange.end = Matrix_::rows;
				if (colRange.start == -1)
					colRange.start = 0;
				if (colRange.end == -1)
                    colRange.end = Matrix_::cols;
				return thrust::make_permutation_iterator(
					operator[](rowRange.start) + colRange.start,
					thrust::make_transform_iterator(
					thrust::make_counting_iterator(0),
                    step_functor(colRange.span(), Matrix_::stride / sizeof(T))));
			}

			iterator end(Range rowRange, Range colRange = Range(-1, -1))
			{
				if (rowRange.start == -1)
					rowRange.start = 0;
				if (rowRange.end == -1)
                    rowRange.end = Matrix_::rows;
				if (colRange.start == -1)
					colRange.start = 0;
				if (colRange.end == -1)
                    colRange.end = Matrix_::cols;
				return thrust::make_permutation_iterator(
					operator[](rowRange.start) + colRange.start,
					thrust::make_transform_iterator(
					thrust::make_counting_iterator(colRange.span()*rowRange.span()),
                    step_functor(colRange.span(), Matrix_::stride / sizeof(T))));
			}
		}; // DeviceMatrix
	} // namespace cuda
} // namespace flann
