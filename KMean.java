import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

public class KMean {

   // user-defined parameters
   private int k;                // number of centroids
   private double[][] points;    // n-dimensional data points. 

   // optional parameters
   private int iterations;      
   private boolean pp;         
   private double epsilon;        
   private boolean useEpsilon;   
                                 
   private boolean L1norm;       // true --> L1 norm to calculate distance; false --> L2 norm

   private int m;              
   private int n;          

   // output
   private double[][] centroids;                      
   private int[] assignment;       
   private double WCSS;         

   // timing information
   private long start;
   private long end;
   

   private KMean() {} 

  
   private KMean(Builder builder) {
      // start timing
      start = System.currentTimeMillis();
      
      // use information from builder
      k = builder.k;
      points = builder.points;
      iterations = builder.iterations;
      pp = builder.pp;
      epsilon = builder.epsilon;
      useEpsilon = builder.useEpsilon;
      L1norm = builder.L1norm;

      // get dimensions to set last 2 fields
      m = points.length;
      n = points[0].length;

      // run KMeans++ clustering algorithm
      run();
      
      end = System.currentTimeMillis();
   }

   public static class Builder {
      // required
      private final int k;
      private final double[][] points;

      // optional (default values given)
      private int iterations     = 10;
      private boolean pp         = true;
      private double epsilon     = .001;
      private boolean useEpsilon = true;
      private boolean L1norm = true;

      public Builder(int k, double[][] points) {
         // check dimensions are valid
         if (k > points.length)
            throw new IllegalArgumentException("Required: # of points >= # of clusters");
         
         HashSet<double[]> hashSet = new HashSet<double[]>(k);
         int distinct = 0;
      
         for (int i = 0; i < points.length; i++) {
            if (!hashSet.contains(points[i])) {
               distinct++;
               if (distinct >= k)
                  break;
               hashSet.add(points[i]);
            }
         }
         
         if (distinct < k)
            throw new IllegalArgumentException("Required: # of distinct points >= # of clusters");
         
         this.k = k;
         this.points = points;
      }

      public Builder iterations(int iterations) {
         if (iterations < 1) 
            throw new IllegalArgumentException("Required: non-negative number of iterations. Ex: 50");
         this.iterations = iterations;
         return this;
      }

      public Builder pp(boolean pp) {
         this.pp = pp;
         return this;
      }

      public Builder epsilon(double epsilon) {
         if (epsilon < 0.0)
            throw new IllegalArgumentException("Required: non-negative value of epsilon. Ex: .001"); 

         this.epsilon = epsilon;
         return this;
      }
      public Builder useEpsilon(boolean useEpsilon) {
         this.useEpsilon = useEpsilon;
         return this;
      }
      
      public Builder useL1norm(boolean L1norm) {
         this.L1norm = L1norm;
         return this;
      }
      public KMean build() {
         return new KMean(this);
      }
   }

   private void run() {
      // for choosing the best run
      double bestWCSS = Double.POSITIVE_INFINITY;
      double[][] bestCentroids = new double[0][0];
      int[] bestAssignment = new int[0];

      // run multiple times and then choose the best run
      for (int n = 0; n < iterations; n++) {
         cluster();

         // store info if it was the best run so far
         if (WCSS < bestWCSS) {
            bestWCSS = WCSS;
            bestCentroids = centroids;
            bestAssignment = assignment;
         }
      }

      // keep info from best run
      WCSS = bestWCSS;
      centroids = bestCentroids;
      assignment = bestAssignment;
   }


   private void cluster() {
      // continue to re-cluster until marginal gains are small enough
      chooseInitialCentroids();
      WCSS = Double.POSITIVE_INFINITY; 
      double prevWCSS;
      do {  
         assignmentStep();   

         updateStep();      

         prevWCSS = WCSS;    
         calcWCSS();
      } while (!stop(prevWCSS));
   }


 
   private void assignmentStep() {
      assignment = new int[m];

      double tempDist;
      double minValue;
      int minLocation;

      for (int i = 0; i < m; i++) {
         minLocation = 0;
         minValue = Double.POSITIVE_INFINITY;
         for (int j = 0; j < k; j++) {
            tempDist = distance(points[i], centroids[j]);
            if (tempDist < minValue) {
               minValue = tempDist;
               minLocation = j;
            }
         }

         assignment[i] = minLocation;
      }

   }



   private void updateStep() {
      // reuse memory is faster than re-allocation
      for (int i = 0; i < k; i++)
         for (int j = 0; j < n; j++)
            centroids[i][j] = 0;
      
      int[] clustSize = new int[k];

      // sum points assigned to each cluster
      for (int i = 0; i < m; i++) {
         clustSize[assignment[i]]++;
         for (int j = 0; j < n; j++)
            centroids[assignment[i]][j] += points[i][j];
      }
      
      // store indices of empty clusters
      HashSet<Integer> emptyCentroids = new HashSet<Integer>();

      // divide to get averages -> centroids
      for (int i = 0; i < k; i++) {
         if (clustSize[i] == 0)
            emptyCentroids.add(i);

         else
            for (int j = 0; j < n; j++)
               centroids[i][j] /= clustSize[i];
      }
      
      // gracefully handle empty clusters by assigning to that centroid an unused data point
      if (emptyCentroids.size() != 0) {
         HashSet<double[]> nonemptyCentroids = new HashSet<double[]>(k - emptyCentroids.size());
         for (int i = 0; i < k; i++)
            if (!emptyCentroids.contains(i))
               nonemptyCentroids.add(centroids[i]);
         
         Random r = new Random();
         for (int i : emptyCentroids)
            while (true) {
               int rand = r.nextInt(points.length);
               if (!nonemptyCentroids.contains(points[rand])) {
                  nonemptyCentroids.add(points[rand]);
                  centroids[i] = points[rand];
                  break;
               }
            }

      }
      
   }

   private void chooseInitialCentroids() {
      if (pp)
         plusplus();
      else
         basicRandSample();
   }

   private void basicRandSample() {
      centroids = new double[k][n];
      double[][] copy = points;

      Random gen = new Random();

      int rand;
      for (int i = 0; i < k; i++) {
         rand = gen.nextInt(m - i);
         for (int j = 0; j < n; j++) {
            centroids[i][j] = copy[rand][j];       // store chosen centroid
            copy[rand][j] = copy[m - 1 - i][j];    // ensure sampling without replacement
         }
      }
   }


   // TODO: see if some of this code is extraneous (can be deleted)
   private void plusplus() {
      centroids = new double[k][n];       
      double[] distToClosestCentroid = new double[m];
      double[] weightedDistribution  = new double[m];  // cumulative sum of squared distances

      Random gen = new Random();
      int choose = 0;

      for (int c = 0; c < k; c++) {

         // first centroid: choose any data point
         if (c == 0)
            choose = gen.nextInt(m);

         // after first centroid, use a weighted distribution
         else {

            // check if the most recently added centroid is closer to any of the points than previously added ones
            for (int p = 0; p < m; p++) {
               // gives chosen points 0 probability of being chosen again -> sampling without replacement
               double tempDistance = Distance.L2(points[p], centroids[c - 1]); // need L2 norm here, not L1

               // base case: if we have only chosen one centroid so far, nothing to compare to
               if (c == 1)
                  distToClosestCentroid[p] = tempDistance;

               else { // c != 1 
                  if (tempDistance < distToClosestCentroid[p])
                     distToClosestCentroid[p] = tempDistance;
               }

               // no need to square because the distance is the square of the euclidean dist
               if (p == 0)
                  weightedDistribution[0] = distToClosestCentroid[0];
               else weightedDistribution[p] = weightedDistribution[p-1] + distToClosestCentroid[p];

            }

            // choose the next centroid
            double rand = gen.nextDouble();
            for (int j = m - 1; j > 0; j--) {
               // TODO: review and try to optimize
               // starts at the largest bin. EDIT: not actually the largest
               if (rand > weightedDistribution[j - 1] / weightedDistribution[m - 1]) { 
                  choose = j; // one bigger than the one above
                  break;
               }
               else // Because of invalid dimension errors, we can't make the forloop go to j2 > -1 when we have (j2-1) in the loop.
                  choose = 0;
            }
         }  

         // store the chosen centroid
         for (int i = 0; i < n; i++)
            centroids[c][i] = points[choose][i];
      }   
   }

   private boolean stop(double prevWCSS) {
      if (useEpsilon)
         return epsilonTest(prevWCSS);
      else
         return prevWCSS == WCSS; // TODO: make comment (more exact, but could be much slower)
      
   }
   private boolean epsilonTest(double prevWCSS) {
      return epsilon > 1 - (WCSS / prevWCSS);
   }

  
   private double distance(double[] x, double[] y) {
      return L1norm ? Distance.L1(x, y) : Distance.L2(x, y);
   }
   
   private static class Distance {
      public static double L1(double[] x, double[] y) {
         if (x.length != y.length) throw new IllegalArgumentException("dimension error");
         double dist = 0;
         for (int i = 0; i < x.length; i++) 
            dist += Math.abs(x[i] - y[i]);
         return dist;
      }
      

      public static double L2(double[] x, double[] y) {
         if (x.length != y.length) throw new IllegalArgumentException("dimension error");
         double dist = 0;
         for (int i = 0; i < x.length; i++)
            dist += Math.abs((x[i] - y[i]) * (x[i] - y[i]));
         return dist;
      }
   }
   
   private void calcWCSS() {
      double WCSS = 0;
      int assignedClust;

      for (int i = 0; i < m; i++) {
         assignedClust = assignment[i];
         WCSS += distance(points[i], centroids[assignedClust]);
      }     

      this.WCSS = WCSS;
   }
   public int[] getAssignment() {
      return assignment;
   }

   public double[][] getCentroids() {
      return centroids;
   }

   public double getWCSS() {
      return WCSS;
   }
   
   public String getTiming() {
      return "KMeans++ took: " + (double) (end - start) / 1000.0 + " seconds";
   }
   public static void main(String args[]) throws IOException {
  
      String data = "TestData.csv";
      int numPoints = 3000;
      int dimensions = 2;
      int k = 4;
      double[][] points = CSVreader.read(data, numPoints, dimensions);

      // run K-means
      final long startTime = System.currentTimeMillis();
      KMean clustering = new KMean.Builder(k, points)
                                    .iterations(50)
                                    .pp(true)
                                    .epsilon(.001)
                                    .useEpsilon(true)
                                    .build();
      final long endTime = System.currentTimeMillis();
     
      // print timing information
      final long elapsed = endTime - startTime;
      System.out.println("Clustering took " + (double) elapsed/1000 + " seconds");
      System.out.println();
      
      // get output
      double[][] centroids = clustering.getCentroids();
      double WCSS          = clustering.getWCSS();
      // int[] assignment  = kmean.getAssignment();

      // print output
      for (int i = 0; i < k; i++)
         System.out.println("(" + centroids[i][0] + ", " + centroids[i][1] + ")");
      System.out.println();

      System.out.println("The within-cluster sum-of-squares (WCSS) = " + WCSS);
      System.out.println();
      
      // write output to CSV
      // CSVwriter.write("filePath", centroids);
   }

}