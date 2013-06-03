package myPolicy;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;
import java.util.Random;

import org.ethz.las.bandit.logs.yahoo.Article;
import org.ethz.las.bandit.logs.yahoo.User;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;
import org.ethz.las.bandit.utils.ArrayHelper;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

public class EfficientHybridPolicy implements
		ContextualBanditPolicy<User, Article, Boolean> {
	private final static int INVERSE_STEPS = 5;
	private final static int SIZE = 6;
	private final static int SIZE2 = SIZE * SIZE;
	private final static double ALFA = 3;
	private final static double EPS = 1e-6;

	private Random random;
	private int inverseSteps;

	private HashMap<Integer, DoubleMatrix> z;
	private HashMap<Integer, DoubleMatrix> A;
	private HashMap<Integer, DoubleMatrix> invA;
	private HashMap<Integer, DoubleMatrix> B;
	private HashMap<Integer, DoubleMatrix> b;

	private DoubleMatrix A0;
	private DoubleMatrix invA0;
	private DoubleMatrix b0;
	
	private DoubleMatrix invAa;
	private DoubleMatrix Ba;
	private DoubleMatrix Ba_tran;
	private DoubleMatrix ba;
	private DoubleMatrix xta;
	private DoubleMatrix xta_tran;
	private DoubleMatrix zta;
	private DoubleMatrix zta_tran;

	// Here you can load the article features.
	public EfficientHybridPolicy(String articleFilePath) {
		random = new Random();
		inverseSteps = 0;

		z = new HashMap<Integer, DoubleMatrix>();
		A = new HashMap<Integer, DoubleMatrix>();
		invA = new HashMap<Integer, DoubleMatrix>();
		B = new HashMap<Integer, DoubleMatrix>();
		b = new HashMap<Integer, DoubleMatrix>();

		A0 = DoubleMatrix.eye(SIZE2);
		invA0 = DoubleMatrix.eye(SIZE2);
		b0 = DoubleMatrix.zeros(SIZE2, 1);

		Scanner scan = null;
		try {
			scan = new Scanner(new File(articleFilePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		scan.useDelimiter("\n");

		while (scan.hasNext()) {
			// Get next line with line scanner.
			String line = scan.next();

			// Tokenize the line.
			String[] tokens = line.split("[\\s]+");

			// Token 0 - is the article ID, tokens 1 - 6 are user features.
			int articleId = Integer.parseInt(tokens[0]);
			double[] features = ArrayHelper.stringArrayToDoubleArray(tokens, 1,
					6);

			z.put(new Integer(articleId), new DoubleMatrix(features));
		}
	}

	@Override
	public Article getActionToPerform(User visitor,
			List<Article> possibleActions) {
		Article maxArticle = null;
		double maxPta = 0;
		int maxCount = 1;

		DoubleMatrix beta = invA0.mmul(b0);

		for (Article article : possibleActions) {
			if (!A.containsKey(article)) {
				A.put(article.getID(), DoubleMatrix.eye(SIZE));
				B.put(article.getID(), DoubleMatrix.zeros(SIZE, SIZE2));
				invA.put(article.getID(), DoubleMatrix.eye(SIZE));
				b.put(article.getID(), DoubleMatrix.zeros(SIZE));
			}

			invAa = invA.get(article.getID());
			Ba = B.get(article.getID());
			Ba_tran = Ba.transpose();
			ba = b.get(article.getID());
			xta = new DoubleMatrix(visitor.getFeatures());
			xta_tran = xta.transpose();
			zta = xta.mmul(z.get(article.getID()).transpose())
					.reshape(SIZE2, 1);
			zta_tran = zta.transpose();

			DoubleMatrix thetaA = invAa.mmul(ba.sub(Ba.mmul(beta)));

			// "Cache" matrices
			DoubleMatrix ztaT_invA0 = zta_tran.mmul(invA0);
			DoubleMatrix xtaT_invAa = xta_tran.mmul(invAa);
			DoubleMatrix BaT_invAa_xta = Ba_tran.mmul(invAa).mmul(xta);

			double sta = ztaT_invA0.mmul(zta).get(0, 0);
			sta -= 2 * ztaT_invA0.mmul(BaT_invAa_xta).get(0, 0);
			sta += xtaT_invAa.mmul(xta).get(0, 0);
			sta += xtaT_invAa.mmul(Ba).mmul(invA0)
					.mmul(BaT_invAa_xta).get(0, 0);

			double pta = zta_tran.mmul(beta).get(0, 0);
			pta += xta_tran.mmul(thetaA).get(0, 0);
			pta += ALFA * Math.sqrt(sta);

			if (pta > maxPta) {
				maxArticle = article;
				maxPta = pta;
				maxCount = 1;
			} else if (Math.abs(maxPta - pta) <= EPS) {
				// Equal
				maxCount++;
				int x = random.nextInt(maxCount);

				if (x == 0) {
					maxArticle = article;
					maxPta = pta;
				}
			}
		}

		return maxArticle;
	}

	@Override
	public void updatePolicy(User c, Article a, Boolean reward) {
		DoubleMatrix Aa = A.get(a.getID());
		
		// "Cache" matrices
		DoubleMatrix BaT_invA = Ba_tran.mmul(invAa);
		DoubleMatrix BaT_invA_Ba = BaT_invA.mmul(Ba);
		DoubleMatrix BaT_invA_ba = BaT_invA.mmul(ba);

		A0 = A0.add(BaT_invA_Ba);
		b0 = b0.add(BaT_invA_ba);
		Aa = Aa.add(xta.mmul(xta_tran));
		Ba = Ba.add(xta.mmul(zta_tran));
		
		// "Cache" matrices
		BaT_invA = Ba_tran.mmul(invAa);
		BaT_invA_Ba = BaT_invA.mmul(Ba);
		
		A0 = A0.add(zta.mmul(zta_tran)).sub(BaT_invA_Ba);
		if (reward) {
			ba = ba.add(xta);
			b0 = b0.add(zta);
		}
		BaT_invA_ba = BaT_invA.mmul(ba);
		b0 = b0.sub(BaT_invA_ba);

		inverseSteps++;
		if (inverseSteps >= INVERSE_STEPS) {
			invA0 = Solve.solve(A0, DoubleMatrix.eye(SIZE2));
			invA.put(a.getID(), Solve.solve(Aa, DoubleMatrix.eye(SIZE)));
			inverseSteps = 0;
		}
	}
}
