using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tracker
{
    public static class MyMath
    {
        public static double GetMean(double[] vec)
        {
            Debug.Assert(vec != null && vec.Length > 0);
            double sum = 0.0;

            foreach(var element in vec)
            {
                sum += element;
            }
            return sum / vec.Length;
        }

        public static double GetStdev(double[] vec)
        {
            Debug.Assert(vec!= null && vec.Length > 0);

            double diffSquareSum = 0.0;
            var mean = GetMean(vec);

            foreach(var element in vec)
            {
                diffSquareSum += Math.Pow((element - mean), 2);
            }

            return Math.Sqrt(diffSquareSum / vec.Length);
        }

        
    }
}
