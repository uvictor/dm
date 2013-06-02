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

public class MyPolicy implements ContextualBanditPolicy<User, Article, Boolean> {
	private final static int SIZE = 6;
	private final static double ALFA = 0.5;
	private final static double EPS = 1e-6;
	
	private Random random;
	
	private HashMap<Integer, DoubleMatrix> x;
	private HashMap<Integer, DoubleMatrix> A;
	private HashMap<Integer, DoubleMatrix> b;

	// Here you can load the article features.
	public MyPolicy(String articleFilePath) {
		random = new Random();
		
		x = new HashMap<Integer, DoubleMatrix>();
		A = new HashMap<Integer, DoubleMatrix>();
		b = new HashMap<Integer, DoubleMatrix>();

		Scanner scan = null;
		try {
			scan = new Scanner(new File(articleFilePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		scan.useDelimiter("\n");

		// Get next line with line scanner.
		String line = scan.next();

		// Tokenize the line.
		String[] tokens = line.split("[\\s]+");

		// Token 0 - is the article ID, tokens 1 - 6 are user features.
		int articleId = Integer.parseInt(tokens[0]);
		double[] features = ArrayHelper.stringArrayToDoubleArray(tokens, 1, 6);

		x.put(new Integer(articleId), new DoubleMatrix(features));
	}

	@Override
	public Article getActionToPerform(User visitor,
			List<Article> possibleActions) {
		Article maxArticle = null;
		double maxUcb = 0;
		int maxCount = 1;
		
		for (Article article: possibleActions) {
			if (!A.containsKey(article)) {
				A.put(article.getID(), DoubleMatrix.eye(SIZE));
				b.put(article.getID(), DoubleMatrix.zeros(SIZE, 1));
			}
			DoubleMatrix Aa = A.get(article.getID());
			DoubleMatrix ba = b.get(article.getID());
			DoubleMatrix zt = new DoubleMatrix(visitor.getFeatures());
			
			DoubleMatrix w = Solve.solve(Aa, ba).transpose();
			
			double ucb = w.mmul(zt).get(0, 0);
			ucb += ALFA * Math.sqrt(zt.transpose().mmul(Solve.solve(Aa, zt)).get(0, 0));
			
			if (ucb > maxUcb) {
				maxArticle = article;
				maxUcb = ucb;
				maxCount = 1;
			} else if (Math.abs(maxUcb - ucb) <= EPS) {
				// Equal
				maxCount++;
				int x = random.nextInt(maxCount);

				if (x == 0) {
					maxArticle = article;
					maxUcb = ucb;
				}
			}
		}
		
		return maxArticle;
	}

	@Override
	public void updatePolicy(User c, Article a, Boolean reward) {
		DoubleMatrix Aa = A.get(a.getID());
		DoubleMatrix ba = b.get(a.getID());
		DoubleMatrix zt = new DoubleMatrix(c.getFeatures());
		
		if (reward) {
			ba = ba.add(zt);
		}
		
		zt = zt.mmul(zt.transpose());
		Aa = Aa.add(zt);
	}
}
