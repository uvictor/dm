package myPolicy;

import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.ethz.las.bandit.logs.yahoo.Article;
import org.ethz.las.bandit.logs.yahoo.User;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

public class DisjointPolicy implements
		ContextualBanditPolicy<User, Article, Boolean> {
	// Algo 1: The time does not lower more if we increase INVERSE_STEPS > 5
	// Algo 1: TIME ~ 11m
	private final static int INVERSE_STEPS = 5;
	private final static int SIZE = 6;
	// Algo 1: Changing ALFA's value does not seem to have considerable effects
	// on the CTR
	private final static double ALFA = 3;
	private final static double EPS = 1e-6;

	private Random random;
	private int inverseSteps;

	private HashMap<Integer, DoubleMatrix> A;
	private HashMap<Integer, DoubleMatrix> invA;
	private HashMap<Integer, DoubleMatrix> b;

	// Here you can load the article features.
	public DisjointPolicy(String articleFilePath) {
		random = new Random();
		inverseSteps = 0;

		A = new HashMap<Integer, DoubleMatrix>();
		invA = new HashMap<Integer, DoubleMatrix>();
		b = new HashMap<Integer, DoubleMatrix>();
	}

	@Override
	public Article getActionToPerform(User visitor,
			List<Article> possibleActions) {
		Article maxArticle = null;
		double maxPta = 0;
		int maxCount = 1;

		for (Article article : possibleActions) {
			if (!A.containsKey(article)) {
				A.put(article.getID(), DoubleMatrix.eye(SIZE));
				invA.put(article.getID(), DoubleMatrix.eye(SIZE));
				b.put(article.getID(), DoubleMatrix.zeros(SIZE, 1));
			}

			DoubleMatrix invAa = invA.get(article.getID());
			DoubleMatrix ba = b.get(article.getID());
			DoubleMatrix xta = new DoubleMatrix(visitor.getFeatures());

			DoubleMatrix thetaA = invAa.mmul(ba).transpose();

			double pta = thetaA.mmul(xta).get(0, 0);
			pta += ALFA
					* Math.sqrt(xta.transpose().mmul(invAa).mmul(xta).get(0, 0));

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
		DoubleMatrix ba = b.get(a.getID());
		DoubleMatrix xta = new DoubleMatrix(c.getFeatures());

		if (reward) {
			ba = ba.add(xta);
		}

		xta = xta.mmul(xta.transpose());
		Aa = Aa.add(xta);

		inverseSteps++;
		if (inverseSteps >= INVERSE_STEPS) {
			invA.put(a.getID(), Solve.solve(Aa, DoubleMatrix.eye(SIZE)));
			inverseSteps = 0;
		}
	}
}
