#ifndef UNIBN_OCTREE_H_
#define UNIBN_OCTREE_H_

// Copyright (c) 2015 Jens Behley, University of Bonn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <vector>
#include <stdint.h>
#include <limits>
#include <cmath>
#include <cassert>
#include <cstring> // memset.
#include <Eigen/Dense>


#include <algorithm>
#include <immintrin.h>

// needed for gtest access to protected/private members ...
namespace
{
	class OctreeTest;
}
class Point3f
{
public:
	Point3f(float x, float y, float z) :
		x(x), y(y), z(z)
	{
	}

	float x, y, z;
};
namespace unibn
{

	/**
	 * Some traits to access coordinates regardless of the specific implementation of point
	 * inspired by boost.geometry, which needs to be implemented by new points.
	 *
	 *//*
	 namespace traits_packed4
	 {

	 template<typename PointT>
	 struct access_pack4
	 {
	 };

	 template<class PointT>
	 struct access_pack4<PointT>
	 {
	 static float* get_all(const PointT& p)
	 {
	 //float
	 return arr[4] = { p.x, p.y, p.z, 0.0f };
	 }
	 };
	 }
	 */

	namespace traits
	{

		template<typename PointT, int D>
		struct access
		{
		};




		template<class PointT>
		struct access<PointT, 0>
		{
			static float get(const PointT& p)
			{
				return p.x;
			}
		};

		template<class PointT>
		struct access<PointT, 1>
		{
			static float get(const PointT& p)
			{
				return p.y;
			}
		};

		template<class PointT>
		struct access<PointT, 2>
		{
			static float get(const PointT& p)
			{
				return p.z;
			}
		};





	}

	/** convenience function for access of point coordinates **/
	template<int D, typename PointT>
	inline float get(const PointT& p)
	{

		return traits::access<PointT, D>::get(p);

	}




	template<typename PointT>
	inline float* get_all_packed4(const PointT& p)
	{
		float arr4[4] = { p.x, p.y, p.z, 0.0f };;
		return arr4;
	}



	/**
	 * Some generic distances: Manhattan, (squared) Euclidean, and Maximum distance.
	 *
	 * A Distance has to implement the methods
	 * 1. compute of two points p and q to compute and return the distance between two points, and
	 * 2. norm of x,y,z coordinates to compute and return the norm of a point p = (x,y,z)
	 * 3. sqr and sqrt of value to compute the correct radius if a comparison is performed using squared norms (see L2Distance)...
	 */
	template<typename PointT>
	struct L1Distance
	{
		static inline float compute(const PointT& p, const PointT& q)
		{
			float diff1 = get<0>(p) -get<0>(q);
			float diff2 = get<1>(p) -get<1>(q);
			float diff3 = get<2>(p) -get<2>(q);

			return std::abs(diff1) + std::abs(diff2) + std::abs(diff3);
		}

		static inline float norm(float x, float y, float z)
		{
			return std::abs(x) + std::abs(y) + std::abs(z);
		}

		static inline float sqr(float r)
		{
			return r;
		}

		static inline float sqrt(float r)
		{
			return r;
		}
	};

	template<typename PointT>
	struct L2Distance
	{
		static inline float compute(const PointT& p, const PointT& q)
		{
			float diff1 = get<0>(p) -get<0>(q);
			float diff2 = get<1>(p) -get<1>(q);
			float diff3 = get<2>(p) -get<2>(q);

			return std::pow(diff1, 2) + std::pow(diff2, 2) + std::pow(diff3, 2);
		}

		static inline float norm(float x, float y, float z)
		{
			return std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2);
		}

		static inline float sqr(float r)
		{
			return r * r;
		}

		static inline float sqrt(float r)
		{
			return std::sqrt(r);
		}
	};

	template<typename PointT>
	struct MaxDistance
	{
		static inline float compute(const PointT& p, const PointT& q)
		{
			float diff1 = std::abs(get<0>(p) -get<0>(q));
			float diff2 = std::abs(get<1>(p) -get<1>(q));
			float diff3 = std::abs(get<2>(p) -get<2>(q));

			float maximum = diff1;
			if (diff2 > maximum) maximum = diff2;
			if (diff3 > maximum) maximum = diff3;

			return maximum;
		}

		static inline float norm(float x, float y, float z)
		{
			float maximum = x;
			if (y > maximum) maximum = y;
			if (z > maximum) maximum = z;
			return maximum;
		}

		static inline float sqr(float r)
		{
			return r;
		}

		static inline float sqrt(float r)
		{
			return r;
		}
	};

	struct OctreeParams
	{
	public:
		OctreeParams(uint32_t bucketSize = 32, bool copyPoints = false)
			: bucketSize(bucketSize), copyPoints(copyPoints)
		{
		}
		uint32_t bucketSize;
		bool copyPoints;
	};

	/** \brief Index-based Octree implementation offering different queries and insertion/removal of points.
	 *
	 * The index-based Octree uses a successor relation and a startIndex in each Octant to improve runtime
	 * performance for radius queries. The efficient storage of the points by relinking list elements
	 * bases on the insight that children of an Octant contain disjoint subsets of points inside the Octant and
	 * that we can reorganize the points such that we get an continuous single connect list that we can use to
	 * store in each octant the start of this list.
	 *
	 * Special about the implementation is that it allows to search for neighbors with arbitrary p-norms, which
	 * distinguishes it from most other Octree implementations.
	 *
	 * We decided to implement the Octree using a template for points and containers. The container must have an
	 * operator[], which allows to access the points, and a size() member function, which allows to get the size of the
	 * container. For the points, we used an access trait to access the coordinates inspired by boost.geometry.
	 * The implementation already provides a general access trait, which expects to have public member variables x,y,z.
	 *
	 * f you use the implementation or ideas from the corresponding paper in your academic work, it would be nice if you
	 * cite the corresponding paper:
	 *
	 *    J. Behley, V. Steinhage, A.B. Cremers. Efficient Radius Neighbor Search in Three-dimensional Point Clouds,
	 *    Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2015.
	 *
	 * In future, we might add also other neighbor queries and implement the removal and adding of points.
	 *
	 * \version 0.1-icra
	 *
	 * \author behley
	 */
	
	 
	 
	

	template<typename PointT, typename ContainerT = std::vector<PointT> >
	class Octree
	{
	public:
		Octree();
		~Octree();

		/** \brief initialize octree with all points **/
		void initialize(const ContainerT& pts, const OctreeParams& params = OctreeParams());

		/** \brief initialize octree only from pts that are inside indexes. **/
		void initialize(const ContainerT& pts, const std::vector<uint32_t>& indexes, const OctreeParams& params =
			OctreeParams());

		/** \brief remove all data inside the octree. **/
		void clear();

		/** \brief radius neighbor queries where radius determines the maximal radius of reported indices of points in resultIndices **/
		template<typename Distance>
		void radiusNeighbors(const PointT& query, float radius, std::vector<uint32_t>& resultIndices) const;

		/** \brief radius neighbor queries with explicit (squared) distance computation. **/
		template<typename Distance>
		void radiusNeighbors(const PointT& query, float radius, std::vector<uint32_t>& resultIndices,
			std::vector<float>& distances) const;

		/** \brief nearest neighbor queries. Using minDistance >= 0, we explicitly disallow self-matches.
		 * @return index of nearest neighbor n with Distance::compute(query, n) > minDistance and otherwise -1.
		 **/
		template<typename Distance>
		int32_t findNeighbor(const PointT& query, float minDistance = -1) const;

	protected:

		class Octant
		{
		public:
			Octant();
			~Octant();

			bool isLeaf;

			// bounding box of the octant needed for overlap and contains tests...
			float x, y, z;    // center
			float extent;    // half of side-length

			uint32_t start, end;    // start and end in succ_
			uint32_t size;    // number of points

			Octant* child[8];
		};

		// not copyable, not assignable ...
		Octree(Octree&);
		Octree& operator=(const Octree& oct);

		/**
		 * \brief creation of an octant using the elements starting at startIdx.
		 *
		 * The method reorders the index such that all points are correctly linked to mSuccessors belonging
		 * to the same octant.
		 *
		 * \param x,y,z           center coordinates of octant
		 * \param extent          extent of octant
		 * \param startIdx        first index of points inside octant
		 * \param endIdx          last index of points inside octant
		 * \param size            number of points in octant
		 *
		 * \return  octant with children nodes.
		 */
		Octant* createOctant(float x, float y, float z, float extent, uint32_t startIdx, uint32_t endIdx, uint32_t size);

		/** @return true, if search finished, otherwise false. **/
		template<typename Distance>
		bool findNeighbor(const Octant* octant, const PointT& query, float minDistance, float& maxDistance,
			int32_t& resultIndex) const;

		template<typename Distance>
		void radiusNeighbors(const Octant* octant, const PointT& query, float radius, float sqrRadius,
			std::vector<uint32_t>& resultIndices) const;

		template<typename Distance>
		void radiusNeighbors(const Octant* octant, const PointT& query, float radius, float sqrRadius,
			std::vector<uint32_t>& resultIndices, std::vector<float>& distances) const;

		/** \brief test if search ball S(q,r) overlaps with octant
		 *
		 * @param query   query point
		 * @param radius  "squared" radius
		 * @param o       pointer to octant
		 *
		 * @return true, if search ball overlaps with octant, false otherwise.
		 */
		template<typename Distance>
		static bool overlaps(const PointT& query, float radius, float sqRadius, const Octant* o);

		template<typename Distance>
		static bool* overlaps_8pack(const PointT& query, float radius, float sqRadius, const Octant* o, const bool *chkNull);


		/** \brief test if search ball S(q,r) contains octant
		 *
		 * @param query    query point
		 * @param sqRadius "squared" radius
		 * @param octant   pointer to octant
		 *
		 * @return true, if search ball overlaps with octant, false otherwise.
		 */
		template<typename Distance>
		static bool contains(const PointT& query, float sqRadius, const Octant* octant);

		/** \brief test if search ball S(q,r) is completely inside octant.
		 *
		 * @param query   query point
		 * @param radius  radius r
		 * @param octant  point to octant.
		 *
		 * @return true, if search ball is completely inside the octant, false otherwise.
		 */
		template<typename Distance>
		static bool inside(const PointT& query, float radius, const Octant* octant);

		OctreeParams params_;
		Octant* root_;
		const ContainerT* data_;

		std::vector<uint32_t> mSuccessors;    // single connected list of next point indices...

		friend class ::OctreeTest;
	};

	template<typename PointT, typename ContainerT>
	Octree<PointT, ContainerT>::Octant::Octant()
		: isLeaf(true), x(0.0f), y(0.0f), z(0.0f), extent(0.0f), start(0), end(0), size(0)
	{
		memset(&child, 0, 8 * sizeof(Octant*));
	}

	template<typename PointT, typename ContainerT>
	Octree<PointT, ContainerT>::Octant::~Octant()
	{
		for (uint32_t i = 0; i < 8; ++i)
			delete child[i];
	}

	template<typename PointT, typename ContainerT>
	Octree<PointT, ContainerT>::Octree()
		: root_(0), data_(0)
	{

	}

	template<typename PointT, typename ContainerT>
	Octree<PointT, ContainerT>::~Octree()
	{
		delete root_;
		if (params_.copyPoints) delete data_;
	}

	template<typename PointT, typename ContainerT>
	void Octree<PointT, ContainerT>::initialize(const ContainerT& pts, const OctreeParams& params)
	{
		clear();
		params_ = params;

		if (params_.copyPoints)
			data_ = new ContainerT(pts);
		else
			data_ = &pts;

		const uint32_t N = pts.size();
		mSuccessors = std::vector<uint32_t>(N);

		// determine axis-aligned bounding box.
		float min[3], max[3];



		min[0] = get<0>(pts[0]);
		min[1] = get<1>(pts[0]);
		min[2] = get<2>(pts[0]);
		//  smd_min(0) = get<0>(pts[0]);
		// simdpp::float32<3> smd_min = simdpp::load(&min);
		//= (, 0.0f);
		// simdpp::float32<3> smd_max = smd_min;


		//__m128 min_mm = (min[0], min[1], min[2], 0.0f);

		max[0] = min[0];
		max[1] = min[1];
		max[2] = min[2];

		for (uint32_t i = 0; i < N; ++i)
		{
			// initially each element links simply to the following element.
			mSuccessors[i] = i + 1;

			const PointT& p = pts[i];
			//simdpp::float32<3> smd_p = simdpp::load(&pts[i]);

			if (get<0>(p) < min[0]) min[0] = get<0>(p);
			if (get<1>(p) < min[1]) min[1] = get<1>(p);
			if (get<2>(p) < min[2]) min[2] = get<2>(p);
			if (get<0>(p) > max[0]) max[0] = get<0>(p);
			if (get<1>(p) > max[1]) max[1] = get<1>(p);
			if (get<2>(p) > max[2]) max[2] = get<2>(p);

			//smd_min = simdpp::min(smd_p, smd_min);
			//smd_max = simdpp::max(smd_p, smd_max);
		}

		float ctr[3] = { min[0], min[1], min[2] };
		//  simdpp::float32<3> smd_ctr = smd_min;
		float maxextent = 0.5f * (max[0] - min[0]);
		ctr[0] += maxextent;
		for (uint32_t i = 1; i < 3; ++i)
		{
			float extent = 0.5f * (max[i] - min[i]);
			ctr[i] += extent;
			if (extent > maxextent) maxextent = extent;
		}

		root_ = createOctant(ctr[0], ctr[1], ctr[2], maxextent, 0, N - 1, N);
	}

	template<typename PointT, typename ContainerT>
	void Octree<PointT, ContainerT>::initialize(const ContainerT& pts, const std::vector<uint32_t>& indexes,
		const OctreeParams& params)
	{
		clear();
		params_ = params;

		if (params_.copyPoints)
			data_ = new ContainerT(pts);
		else
			data_ = &pts;

		const uint32_t N = pts.size();
		mSuccessors = std::vector<uint32_t>(N);

		if (indexes.size() == 0) return;

		// determine axis-aligned bounding box.
		uint32_t lastIdx = indexes[0];
		float min[3], max[3];
		min[0] = get<0>(pts[lastIdx]);
		min[1] = get<1>(pts[lastIdx]);
		min[2] = get<2>(pts[lastIdx]);
		max[0] = min[0];
		max[1] = min[1];
		max[2] = min[2];

		for (uint32_t i = 1; i < indexes.size(); ++i)
		{
			uint32_t idx = indexes[i];
			// initially each element links simply to the following element.
			mSuccessors[lastIdx] = idx;

			const PointT& p = pts[idx];

			if (get<0>(p) < min[0]) min[0] = get<0>(p);
			if (get<1>(p) < min[1]) min[1] = get<1>(p);
			if (get<2>(p) < min[2]) min[2] = get<2>(p);
			if (get<0>(p) > max[0]) max[0] = get<0>(p);
			if (get<1>(p) > max[1]) max[1] = get<1>(p);
			if (get<2>(p) > max[2]) max[2] = get<2>(p);

			lastIdx = idx;
		}

		float ctr[3] = { min[0], min[1], min[2] };

		float maxextent = 0.5f * (max[0] - min[0]);
		ctr[0] += maxextent;
		for (uint32_t i = 1; i < 3; ++i)
		{
			float extent = 0.5f * (max[i] - min[i]);
			ctr[i] += extent;
			if (extent > maxextent) maxextent = extent;
		}

		root_ = createOctant(ctr[0], ctr[1], ctr[2], maxextent, indexes[0], lastIdx, indexes.size());
	}

	template<typename PointT, typename ContainerT>
	void Octree<PointT, ContainerT>::clear()
	{
		delete root_;
		if (params_.copyPoints) delete data_;
		root_ = 0;
		data_ = 0;
		mSuccessors.clear();
	}

	template<typename PointT, typename ContainerT>
	typename Octree<PointT, ContainerT>::Octant* Octree<PointT, ContainerT>::createOctant(float x, float y, float z,
		float extent, uint32_t startIdx, uint32_t endIdx, uint32_t size)
	{
		// For a leaf we don't have to change anything; points are already correctly linked or correctly reordered.
		Octant* octant = new Octant;

		octant->isLeaf = true;

		octant->x = x;
		octant->y = y;
		octant->z = z;
		octant->extent = extent;

		octant->start = startIdx;
		octant->end = endIdx;
		octant->size = size;

		static const float factor[] = { -0.5f, 0.5f };

		// subdivide subset of points and re-link points according to Morton codes
		if (size > params_.bucketSize)
		{
			octant->isLeaf = false;

			const ContainerT& points = *data_;
			std::vector<uint32_t> childStarts(8, 0);
			std::vector<uint32_t> childEnds(8, 0);
			std::vector<uint32_t> childSizes(8, 0);

			// re-link disjoint child subsets...
			uint32_t idx = startIdx;

			for (uint32_t i = 0; i < size; ++i)
			{
				const PointT& p = points[idx];

				// determine Morton code for each point...
				uint32_t mortonCode = 0;
				if (get<0>(p) > x) mortonCode |= 1;
				if (get<1>(p) > y) mortonCode |= 2;
				if (get<2>(p) > z) mortonCode |= 4;

				// set child starts and update mSuccessors...
				if (childSizes[mortonCode] == 0)
					childStarts[mortonCode] = idx;
				else
					mSuccessors[childEnds[mortonCode]] = idx;

				childSizes[mortonCode] += 1;

				childEnds[mortonCode] = idx;
				idx = mSuccessors[idx];
			}

			// now, we can create the child nodes...
			float childExtent = 0.5f * extent;
			bool firsttime = true;

			uint32_t lastChildIdx = 0;
			for (uint32_t i = 0; i < 8; ++i)
			{
				if (childSizes[i] == 0) continue;

				float childX = x + factor[(i & 1) > 0] * extent;
				float childY = y + factor[(i & 2) > 0] * extent;
				float childZ = z + factor[(i & 4) > 0] * extent;

				octant->child[i] = createOctant(childX, childY, childZ, childExtent, childStarts[i], childEnds[i], childSizes[i]);

				if (firsttime)
					octant->start = octant->child[i]->start;
				else
					mSuccessors[octant->child[lastChildIdx]->end] = octant->child[i]->start;    // we have to ensure that also the child ends link to the next child start.

				lastChildIdx = i;
				octant->end = octant->child[i]->end;
				firsttime = false;
			}
		}

		return octant;
	}

	template<typename PointT, typename ContainerT>
	template<typename Distance>
	void Octree<PointT, ContainerT>::radiusNeighbors(const Octant* octant, const PointT& query, float radius,
		float sqrRadius, std::vector<uint32_t>& resultIndices) const
	{
		const ContainerT& points = *data_;

		// if search ball S(q,r) contains octant, simply add point indexes.
		if (contains<Distance>(query, sqrRadius, octant))
		{
			uint32_t idx = octant->start;
			for (uint32_t i = 0; i < octant->size; ++i)
			{
				resultIndices.push_back(idx);
				idx = mSuccessors[idx];
			}

			return;    // early pruning.
		}

		if (octant->isLeaf)
		{
			uint32_t idx = octant->start;
			for (uint32_t i = 0; i < octant->size; ++i)
			{
				const PointT& p = points[idx];
				float dist = Distance::compute(query, p);
				if (dist < sqrRadius) resultIndices.push_back(idx);
				idx = mSuccessors[idx];
			}

			return;
		}

		// check whether child nodes are in range.
		for (uint32_t c = 0; c < 8; ++c)
		{
			if (octant->child[c] == 0) continue;
			if (!overlaps<Distance>(query, radius, sqrRadius, octant->child[c])) continue;
			radiusNeighbors<Distance>(octant->child[c], query, radius, sqrRadius, resultIndices);
		}
	}

	template<typename PointT, typename ContainerT>
	template<typename Distance>
	void Octree<PointT, ContainerT>::radiusNeighbors(const Octant* octant, const PointT& query, float radius,
		float sqrRadius, std::vector<uint32_t>& resultIndices, std::vector<float>& distances) const
	{
		const ContainerT& points = *data_;

		// if search ball S(q,r) contains octant, simply add point indexes and compute squared distances.
		if (contains<Distance>(query, sqrRadius, octant))
		{
			uint32_t idx = octant->start;
			for (uint32_t i = 0; i < octant->size; ++i)
			{
				resultIndices.push_back(idx);
				distances.push_back(Distance::compute(query, points[idx]));
				idx = mSuccessors[idx];
			}

			return;    // early pruning.
		}

		if (octant->isLeaf)
		{
			uint32_t idx = octant->start;
			for (uint32_t i = 0; i < octant->size; ++i)
			{
				const PointT& p = points[idx];
				float dist = Distance::compute(query, p);
				if (dist < sqrRadius)
				{
					resultIndices.push_back(idx);
					distances.push_back(dist);
				}
				idx = mSuccessors[idx];
			}

			return;
		}

		// check whether child nodes are in range.
		for (uint32_t c = 0; c < 8; ++c)
		{
			if (octant->child[c] == 0) continue;
			if (!overlaps<Distance>(query, radius, sqrRadius, octant->child[c])) continue;
			radiusNeighbors<Distance>(octant->child[c], query, radius, sqrRadius, resultIndices, distances);
		}
	}

	template<typename PointT, typename ContainerT>
	template<typename Distance>
	void Octree<PointT, ContainerT>::radiusNeighbors(const PointT& query, float radius,
		std::vector<uint32_t>& resultIndices) const
	{
		resultIndices.clear();
		if (root_ == 0) return;

		float sqrRadius = Distance::sqr(radius);    // "squared" radius
		radiusNeighbors<Distance>(root_, query, radius, sqrRadius, resultIndices);
	}

	template<typename PointT, typename ContainerT>
	template<typename Distance>
	void Octree<PointT, ContainerT>::radiusNeighbors(const PointT& query, float radius,
		std::vector<uint32_t>& resultIndices, std::vector<float>& distances) const
	{
		resultIndices.clear();
		distances.clear();
		if (root_ == 0) return;

		float sqrRadius = Distance::sqr(radius);    // "squared" radius
		radiusNeighbors<Distance>(root_, query, radius, sqrRadius, resultIndices, distances);
	}

	template<typename PointT, typename ContainerT>
	template<typename Distance>
	bool Octree<PointT, ContainerT>::overlaps(const PointT& query, float radius, float sqRadius, const Octant* o)
	{
		// we exploit the symmetry to reduce the test to testing if its inside the Minkowski sum around the positive quadrant.

		float x = get<0>(query) -o->x;
		float y = get<1>(query) -o->y;
		float z = get<2>(query) -o->z;

		x = std::abs(x);
		y = std::abs(y);
		z = std::abs(z);

		// (1) checking the line region.
		float maxdist = radius + o->extent;

		// a. completely outside, since q' is outside the relevant area.
		if (x > maxdist || y > maxdist || z > maxdist) return false;

		// b. inside the line region, one of the coordinates is inside the square.
		if (x < o->extent || y < o->extent || z < o->extent) return true;

		// (2) checking the corner region...
		x -= o->extent;
		y -= o->extent;
		z -= o->extent;

		return (Distance::norm(x, y, z) < sqRadius);



	}

	template<typename PointT, typename ContainerT>
	template<typename Distance>
	bool Octree<PointT, ContainerT>::contains(const PointT& query, float sqRadius, const Octant* o)
	{
		// we exploit the symmetry to reduce the test to test
		// whether the farthest corner is inside the search ball.
		float x = get<0>(query) -o->x;
		float y = get<1>(query) -o->y;
		float z = get<2>(query) -o->z;

		x = std::abs(x);
		y = std::abs(y);
		z = std::abs(z);
		// reminder: (x, y, z) - (-e, -e, -e) = (x, y, z) + (e, e, e)
		x += o->extent;
		y += o->extent;
		z += o->extent;

		return (Distance::norm(x, y, z) < sqRadius);
	}

	template<typename PointT, typename ContainerT>
	template<typename Distance>
	int32_t Octree<PointT, ContainerT>::findNeighbor(const PointT& query, float minDistance) const
	{
		float maxDistance = std::numeric_limits<float>::infinity();
		int32_t resultIndex = -1;
		findNeighbor<Distance>(root_, query, minDistance, maxDistance, resultIndex);

		return resultIndex;
	}
#define avx 1
	template<typename PointT, typename ContainerT>
	template<typename Distance>
	bool Octree<PointT, ContainerT>::findNeighbor(const Octant* octant, const PointT& query, float minDistance,
		float& maxDistance, int32_t& resultIndex) const
	{
		const ContainerT& points = *data_;


		//	float query_arr[3] = (get<2>(query), get<1>(query), get<>(query));
		// 1. first descend to leaf and check in leafs points.

	//	std::cout << "IS LEAF : " << octant->isLeaf << " , " << "octant size : " << octant->size << std::endl;

		__m256 vec_query = _mm256_set_ps(0, get<2>(query), get<1>(query), get<0>(query), 0, get<2>(query), get<1>(query), get<0>(query));

		if (octant->isLeaf)
		{


			uint32_t idx1 = octant->start;

			float sqrMaxDistance = Distance::sqr(maxDistance);
			float sqrMinDistance = (minDistance < 0) ? minDistance : Distance::sqr(minDistance);
			float dist_arr[8];

#if avx==0


			//std::cout << "octant size " << octant->size << std::endl;

			for (uint32_t i = 0; i < octant->size; ++i)

			{
				const PointT& p = points[idx1];
				float dist = Distance::compute(query, p);
				if (dist > sqrMinDistance && dist < sqrMaxDistance)
				{
					resultIndex = idx1;
					sqrMaxDistance = dist;
				}
				idx1 = mSuccessors[idx1];
			}

#elif avx==1

			/**
			* avx sse 로 가속화한 코드
			* successor는 index정보를 가지고 있는 컨테이너이며 링크드리스트처럼 구성됨
			* successor를 따라가면서 point 정보를 가져옴
			* 이를 기준점(query)으로부터 유클리드 거리계산
			*
			* simd로 가속화 하기위해 successor에서 index값 빼낸 후 처리
			*
			*/


			float zero_value = 0.0f;
			__m256 vec_zero = _mm256_set1_ps(zero_value);
			__m128 vec_x_query_128 = _mm_set1_ps(get<0>(query));
			__m128 vec_y_query_128 = _mm_set1_ps(get<1>(query));
			__m128 vec_z_query_128 = _mm_set1_ps(get<2>(query));


			__m256 vec_x_query = _mm256_set_m128(vec_x_query_128, vec_x_query_128);
			__m256 vec_y_query = _mm256_set_m128(vec_y_query_128, vec_y_query_128);
			__m256 vec_z_query = _mm256_set_m128(vec_z_query_128, vec_z_query_128);


			int iter_asc = octant->size;

			//링크드리스트 형태이므로 이전의 값을 이용해 인덱스 값 불러옴
			uint32_t idx2 = mSuccessors[idx1];
			uint32_t idx3 = mSuccessors[idx2];
			uint32_t idx4 = mSuccessors[idx3];
			uint32_t idx5 = mSuccessors[idx4];
			uint32_t idx6 = mSuccessors[idx5];
			uint32_t idx7 = mSuccessors[idx6];
			uint32_t idx8 = mSuccessors[idx7];

			int counter = 0;
			for (uint32_t i = iter_asc; i > 0; i -= counter)

			{
				// octant(박스) 내부에 있는 점의 개수가 8개 이상일 때
				// 6 4 2 동일하게 simd 처리 
				if (i / 8 >= 1)
				{
					counter = 8;
					__m256 vec8_x = _mm256_set_ps(
						get<0>(points[idx1]),
						get<0>(points[idx2]),
						get<0>(points[idx3]),
						get<0>(points[idx4]),
						get<0>(points[idx5]),
						get<0>(points[idx6]),
						get<0>(points[idx7]),
						get<0>(points[idx8]));
					__m256 vec8_y = _mm256_set_ps(
						get<1>(points[idx1]),
						get<1>(points[idx2]),
						get<1>(points[idx3]),
						get<1>(points[idx4]),
						get<1>(points[idx5]),
						get<1>(points[idx6]),
						get<1>(points[idx7]),
						get<1>(points[idx8]));
					__m256 vec8_z = _mm256_set_ps(
						get<2>(points[idx1]),
						get<2>(points[idx2]),
						get<2>(points[idx3]),
						get<2>(points[idx4]),
						get<2>(points[idx5]),
						get<2>(points[idx6]),
						get<2>(points[idx7]),
						get<2>(points[idx8]));

					// x y z 각각의 값에 대한 차이
					__m256 vec_diff_x = _mm256_sub_ps(vec_x_query, vec8_x);
					__m256 vec_diff_y = _mm256_sub_ps(vec_y_query, vec8_y);
					__m256 vec_diff_z = _mm256_sub_ps(vec_z_query, vec8_z);

					//제곱
					vec_diff_x = _mm256_mul_ps(vec_diff_x, vec_diff_x);
					vec_diff_y = _mm256_mul_ps(vec_diff_y, vec_diff_y);
					vec_diff_z = _mm256_mul_ps(vec_diff_z, vec_diff_z);


					// x^2 + y^2 + z^2
					__m256 vec_sum = _mm256_add_ps(vec_diff_x, vec_diff_y);
					vec_sum = _mm256_add_ps(vec_sum, vec_diff_z);
					_mm256_store_ps(&dist_arr[0], vec_sum);





					if (dist_arr[0] > sqrMinDistance && dist_arr[0] < sqrMaxDistance) //2
					{
						resultIndex = idx8;
						sqrMaxDistance = dist_arr[0];
					}
					if (dist_arr[1] > sqrMinDistance && dist_arr[1] < sqrMaxDistance) //2
					{
						resultIndex = idx7;
						sqrMaxDistance = dist_arr[1];
					}
					if (dist_arr[2] > sqrMinDistance && dist_arr[2] < sqrMaxDistance) //2
					{
						resultIndex = idx6;
						sqrMaxDistance = dist_arr[2];
					}
					if (dist_arr[3] > sqrMinDistance && dist_arr[3] < sqrMaxDistance) //2
					{
						resultIndex = idx5;
						sqrMaxDistance = dist_arr[3];
					}
					if (dist_arr[4] > sqrMinDistance && dist_arr[4] < sqrMaxDistance) //2
					{
						resultIndex = idx4;
						sqrMaxDistance = dist_arr[4];
					}
					if (dist_arr[5] > sqrMinDistance && dist_arr[5] < sqrMaxDistance) //2
					{
						resultIndex = idx3;
						sqrMaxDistance = dist_arr[5];
					}
					if (dist_arr[6] > sqrMinDistance && dist_arr[6] < sqrMaxDistance) //2
					{
						resultIndex = idx2;
						sqrMaxDistance = dist_arr[6];
					}
					if (dist_arr[7] > sqrMinDistance && dist_arr[7] < sqrMaxDistance) //2
					{
						resultIndex = idx1;
						sqrMaxDistance = dist_arr[7];
					}



					// sqrmaxdistance값과 distance값 비교하여 최소값으로 업데이트



					idx1 = mSuccessors[idx8];
					idx2 = mSuccessors[idx1];
					idx3 = mSuccessors[idx2];
					idx4 = mSuccessors[idx3];

					idx5 = mSuccessors[idx4];
					idx6 = mSuccessors[idx5];
					idx7 = mSuccessors[idx6];
					idx8 = mSuccessors[idx7];




				}
				/*
				else if (i / 7 >= 1)
				{
					counter = 7;
					__m256 vec8_x = _mm256_set_ps(
						get<0>(points[idx1]),
						get<0>(points[idx2]),
						get<0>(points[idx3]),
						get<0>(points[idx4]),
						get<0>(points[idx5]),
						get<0>(points[idx6]),
						get<0>(points[idx7]),
						0);
					__m256 vec8_y = _mm256_set_ps(
						get<1>(points[idx1]),
						get<1>(points[idx2]),
						get<1>(points[idx3]),
						get<1>(points[idx4]),
						get<1>(points[idx5]),
						get<1>(points[idx6]),
						get<1>(points[idx7]),
						0);
					__m256 vec8_z = _mm256_set_ps(
						get<2>(points[idx1]),
						get<2>(points[idx2]),
						get<2>(points[idx3]),
						get<2>(points[idx4]),
						get<2>(points[idx5]),
						get<2>(points[idx6]),
						get<2>(points[idx7]),
						0);

					// x y z 각각의 값에 대한 차이
					__m256 vec_diff_x = _mm256_sub_ps(vec_x_query, vec8_x);
					__m256 vec_diff_y = _mm256_sub_ps(vec_y_query, vec8_y);
					__m256 vec_diff_z = _mm256_sub_ps(vec_z_query, vec8_z);

					//제곱
					vec_diff_x = _mm256_mul_ps(vec_diff_x, vec_diff_x);
					vec_diff_y = _mm256_mul_ps(vec_diff_y, vec_diff_y);
					vec_diff_z = _mm256_mul_ps(vec_diff_z, vec_diff_z);


					// x^2 + y^2 + z^2
					__m256 vec_sum = _mm256_add_ps(vec_diff_x, vec_diff_y);
					vec_sum = _mm256_add_ps(vec_sum, vec_diff_z);
					_mm256_store_ps(&dist_arr[0], vec_sum);






					if (dist_arr[1] > sqrMinDistance && dist_arr[1] < sqrMaxDistance) //2
					{
						resultIndex = idx7;
						sqrMaxDistance = dist_arr[1];
					}
					if (dist_arr[2] > sqrMinDistance && dist_arr[2] < sqrMaxDistance) //2
					{
						resultIndex = idx6;
						sqrMaxDistance = dist_arr[2];
					}
					if (dist_arr[3] > sqrMinDistance && dist_arr[3] < sqrMaxDistance) //2
					{
						resultIndex = idx5;
						sqrMaxDistance = dist_arr[3];
					}
					if (dist_arr[4] > sqrMinDistance && dist_arr[4] < sqrMaxDistance) //2
					{
						resultIndex = idx4;
						sqrMaxDistance = dist_arr[4];
					}
					if (dist_arr[5] > sqrMinDistance && dist_arr[5] < sqrMaxDistance) //2
					{
						resultIndex = idx3;
						sqrMaxDistance = dist_arr[5];
					}
					if (dist_arr[6] > sqrMinDistance && dist_arr[6] < sqrMaxDistance) //2
					{
						resultIndex = idx2;
						sqrMaxDistance = dist_arr[6];
					}
					if (dist_arr[7] > sqrMinDistance && dist_arr[7] < sqrMaxDistance) //2
					{
						resultIndex = idx1;
						sqrMaxDistance = dist_arr[7];
					}



					// sqrmaxdistance값과 distance값 비교하여 최소값으로 업데이트



					idx1 = mSuccessors[idx7];
					idx2 = mSuccessors[idx1];
					idx3 = mSuccessors[idx2];
					idx4 = mSuccessors[idx3];

					idx5 = mSuccessors[idx4];
					idx6 = mSuccessors[idx5];
					idx7 = mSuccessors[idx6];




				}
				*/
				else if (i / 6 >= 1)
				{
					counter = 6;

					__m256 vec6_x = _mm256_set_ps(
						get<0>(points[idx1]),
						get<0>(points[idx2]),
						get<0>(points[idx3]),
						get<0>(points[idx4]),
						get<0>(points[idx5]),
						get<0>(points[idx6]),
						0, 0);
					__m256 vec6_y = _mm256_set_ps(
						get<1>(points[idx1]),
						get<1>(points[idx2]),
						get<1>(points[idx3]),
						get<1>(points[idx4]),
						get<1>(points[idx5]),
						get<1>(points[idx6]),
						0, 0);
					__m256 vec6_z = _mm256_set_ps(
						get<2>(points[idx1]),
						get<2>(points[idx2]),
						get<2>(points[idx3]),
						get<2>(points[idx4]),
						get<2>(points[idx5]),
						get<2>(points[idx6]),
						0, 0);

					__m256 vec_diff_x = _mm256_sub_ps(vec_x_query, vec6_x);
					__m256 vec_diff_y = _mm256_sub_ps(vec_y_query, vec6_y);
					__m256 vec_diff_z = _mm256_sub_ps(vec_z_query, vec6_z);

					vec_diff_x = _mm256_mul_ps(vec_diff_x, vec_diff_x);
					vec_diff_y = _mm256_mul_ps(vec_diff_y, vec_diff_y);
					vec_diff_z = _mm256_mul_ps(vec_diff_z, vec_diff_z);

					__m256 vec_sum = _mm256_add_ps(vec_diff_x, vec_diff_y);
					vec_sum = _mm256_add_ps(vec_sum, vec_diff_z);
					_mm256_store_ps(&dist_arr[0], vec_sum);




					_mm256_store_ps(&dist_arr[0], vec_sum);

					if (dist_arr[2] > sqrMinDistance && dist_arr[2] < sqrMaxDistance) //2
					{
						resultIndex = idx6;
						sqrMaxDistance = dist_arr[2];
					}
					if (dist_arr[3] > sqrMinDistance && dist_arr[3] < sqrMaxDistance) //2
					{
						resultIndex = idx5;
						sqrMaxDistance = dist_arr[3];
					}
					if (dist_arr[4] > sqrMinDistance && dist_arr[4] < sqrMaxDistance) //2
					{
						resultIndex = idx4;
						sqrMaxDistance = dist_arr[4];
					}
					if (dist_arr[5] > sqrMinDistance && dist_arr[5] < sqrMaxDistance) //2
					{
						resultIndex = idx3;
						sqrMaxDistance = dist_arr[5];
					}
					if (dist_arr[6] > sqrMinDistance && dist_arr[6] < sqrMaxDistance) //2
					{
						resultIndex = idx2;
						sqrMaxDistance = dist_arr[6];
					}
					if (dist_arr[7] > sqrMinDistance && dist_arr[7] < sqrMaxDistance) //2
					{
						resultIndex = idx1;
						sqrMaxDistance = dist_arr[7];
					}
					idx1 = mSuccessors[idx6];
					idx2 = mSuccessors[idx1];
					idx3 = mSuccessors[idx2];
					idx4 = mSuccessors[idx3];

					idx5 = mSuccessors[idx4];
					idx6 = mSuccessors[idx5];

				}
				else if (i / 4 >= 1)
				{
					counter = 4;

					__m256 vec_12 = _mm256_set_ps
						(0, get<2>(points[idx2]), get<1>(points[idx2]), get<0>(points[idx2]),
						0, get<2>(points[idx1]), get<1>(points[idx1]), get<0>(points[idx1]));

					__m256 vec_34 = _mm256_set_ps
						(0, get<2>(points[idx4]), get<1>(points[idx4]), get<0>(points[idx4]),
						0, get<2>(points[idx3]), get<1>(points[idx3]), get<0>(points[idx3]));

					__m256 vec_diff1 = _mm256_sub_ps(vec_query, vec_12);
					__m256 vec_diff2 = _mm256_sub_ps(vec_query, vec_34);


					vec_diff1 = _mm256_mul_ps(vec_diff1, vec_diff1);
					vec_diff2 = _mm256_mul_ps(vec_diff2, vec_diff2);
					//	_vec_diff= _mm256_add_ps()
					vec_diff1 = _mm256_hadd_ps(vec_diff1, vec_diff2);//0 0 z x+y 0 0 z x+y
					vec_diff1 = _mm256_hadd_ps(vec_diff1, vec_zero);// 0 0 0 x+y+z 0 0 0 x+y+z

					_mm256_store_ps(&dist_arr[0], vec_diff1);




					if (dist_arr[0] > sqrMinDistance && dist_arr[0] < sqrMaxDistance) //2
					{
						resultIndex = idx1;
						sqrMaxDistance = dist_arr[0];
					}
					if (dist_arr[1] > sqrMinDistance && dist_arr[1] < sqrMaxDistance) //2
					{
						resultIndex = idx3;
						sqrMaxDistance = dist_arr[1];
					}
					if (dist_arr[4] > sqrMinDistance && dist_arr[4] < sqrMaxDistance) //2
					{
						resultIndex = idx2;
						sqrMaxDistance = dist_arr[4];
					}
					if (dist_arr[5] > sqrMinDistance && dist_arr[5] < sqrMaxDistance) //2
					{
						resultIndex = idx4;
						sqrMaxDistance = dist_arr[5];
					}

					idx1 = mSuccessors[idx4];
					idx2 = mSuccessors[idx1];
					idx3 = mSuccessors[idx2];
					idx4 = mSuccessors[idx3];




					idx1 = mSuccessors[idx4];
					idx2 = mSuccessors[idx1];

					idx3 = mSuccessors[idx2];
					idx4 = mSuccessors[idx3];






				}
				else if (i / 2 >= 1)
				{
					counter = 2;

					__m256 vec_12 = _mm256_set_ps
						(0, get<2>(points[idx2]), get<1>(points[idx2]), get<0>(points[idx2]),
						0, get<2>(points[idx1]), get<1>(points[idx1]), get<0>(points[idx1]));

					__m256 vec_diff = _mm256_sub_ps(vec_query, vec_12);


					vec_diff = _mm256_mul_ps(vec_query, vec_12);

					//	_vec_diff= _mm256_add_ps()


					vec_diff = _mm256_hadd_ps(vec_diff, vec_zero);//0 0 z x+y 0 0 z x+y
					vec_diff = _mm256_hadd_ps(vec_diff, vec_zero);// 0 0 0 x+y+z 0 0 0 x+y+z
					_mm256_store_ps(&dist_arr[0], vec_diff);


					if (dist_arr[0] > sqrMinDistance && dist_arr[0] < sqrMaxDistance) //2
					{
						resultIndex = idx1;
						sqrMaxDistance = dist_arr[0];
					}
					if (dist_arr[4] > sqrMinDistance && dist_arr[4] < sqrMaxDistance) //2
					{
						resultIndex = idx2;
						sqrMaxDistance = dist_arr[4];
					}

					idx1 = mSuccessors[idx2];
					idx2 = mSuccessors[idx1];




				}
				else
				{
					counter = 1;
					const PointT& p = points[idx1];
					float dist = Distance::compute(query, p);
					if (dist > sqrMinDistance && dist < sqrMaxDistance)
					{
						resultIndex = idx1;
						sqrMaxDistance = dist;
					}
					idx1 = mSuccessors[idx1];
				}

			}
#endif
			maxDistance = Distance::sqrt(sqrMaxDistance);

			return inside<Distance>(query, maxDistance, octant);
		}

		// determine Morton code for each point...
		uint32_t mortonCode = 0;



		if (get<0>(query) > octant->x) mortonCode |= 1;
		if (get<1>(query) > octant->y) mortonCode |= 2;
		if (get<2>(query) > octant->z) mortonCode |= 4;



		if (octant->child[mortonCode] != 0)
		{
			if (findNeighbor<Distance>(octant->child[mortonCode], query, minDistance, maxDistance, resultIndex)) return true;
		}


		// 2. if current best point completely inside, just return.
		float sqrMaxDistance = Distance::sqr(maxDistance);


		// 3. check adjacent octants for overlap and check these if necessary.

		for (uint32_t c = 0; c < 8; ++c)
		{
			if (c == mortonCode) continue;
			if (octant->child[c] == 0) continue;

			if (!overlaps<Distance>(query, maxDistance, sqrMaxDistance, octant->child[c])) continue;
			if (findNeighbor<Distance>(octant->child[c], query, minDistance, maxDistance, resultIndex)) return true;    // early pruning

		}







		return inside<Distance>(query, maxDistance, octant);

	}

	template<typename PointT, typename ContainerT>
	template<typename Distance>
	bool Octree<PointT, ContainerT>::inside(const PointT& query, float radius, const Octant* octant)
	{
		// we exploit the symmetry to reduce the test to test
		// whether the farthest corner is inside the search ball.

		float x = get<0>(query) -octant->x;
		float y = get<1>(query) -octant->y;
		float z = get<2>(query) -octant->z;

		x = std::abs(x) + radius;
		y = std::abs(y) + radius;
		z = std::abs(z) + radius;

		if (x > octant->extent) return false;
		if (y > octant->extent) return false;
		if (z > octant->extent) return false;


		return true;
	}



}
#endif /* OCTREE_HPP_ */