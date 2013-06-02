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

public class HybridPolicy implements
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

	// Here you can load the article features.
	public HybridPolicy(String articleFilePath) {
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

			DoubleMatrix invAa = invA.get(article.getID());
			DoubleMatrix Ba = B.get(article.getID());
			DoubleMatrix Ba_tran = Ba.transpose();
			DoubleMatrix ba = b.get(article.getID());
			DoubleMatrix xta = new DoubleMatrix(visitor.getFeatures());
			DoubleMatrix xta_tran = xta.transpose();
			DoubleMatrix zta = xta.mmul(z.get(article.getID()).transpose())
					.reshape(SIZE2, 1);
			DoubleMatrix zta_tran = zta.transpose();

			DoubleMatrix thetaA = invAa.mmul(ba.sub(Ba.mmul(beta)));

			double sta = zta_tran.mmul(invA0).mmul(zta).get(0, 0);
			sta -= 2 * zta_tran.mmul(invA0).mmul(Ba_tran).mmul(invAa).mmul(xta)
					.get(0, 0);
			sta += xta_tran.mmul(invAa).mmul(xta).get(0, 0);
			sta += xta_tran.mmul(invAa).mmul(Ba).mmul(invA0).mmul(Ba_tran)
					.mmul(invAa).mmul(xta).get(0, 0);

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
		DoubleMatrix invAa = invA.get(a.getID());
		DoubleMatrix Ba = B.get(a.getID());
		DoubleMatrix Ba_tran = Ba.transpose();
		DoubleMatrix ba = b.get(a.getID());
		DoubleMatrix xta = new DoubleMatrix(c.getFeatures());
		DoubleMatrix xta_tran = xta.transpose();
		DoubleMatrix zta = xta.mmul(z.get(a.getID()).transpose()).reshape(
				SIZE2, 1);
		DoubleMatrix zta_tran = zta.transpose();

		A0 = A0.add(Ba_tran.mmul(invAa).mmul(Ba));
		b0 = b0.add(Ba_tran.mmul(invAa).mmul(ba));
		Aa = Aa.add(xta.mmul(xta_tran));
		Ba = Ba.add(xta.mmul(zta_tran));
		A0 = A0.add(zta.mmul(zta_tran)).sub(Ba_tran.mmul(invAa).mmul(Ba));

		if (reward) {
			ba = ba.add(xta);
			b0 = b0.add(zta);
		}
		b0 = b0.sub(Ba_tran.mmul(invAa).mmul(ba));

		inverseSteps++;
		if (inverseSteps >= INVERSE_STEPS) {
			invA0 = Solve.solve(A0, DoubleMatrix.eye(SIZE2));
			invA.put(a.getID(), Solve.solve(Aa, DoubleMatrix.eye(SIZE)));
			inverseSteps = 0;
		}
	}
}
