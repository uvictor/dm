package myPolicy;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

import org.ethz.las.bandit.logs.yahoo.Article;
import org.ethz.las.bandit.logs.yahoo.ArticleFeatures;
import org.ethz.las.bandit.logs.yahoo.User;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;
import org.ethz.las.bandit.utils.ArrayHelper;

public class MyPolicy implements ContextualBanditPolicy<User, Article, Boolean> {
	private HashSet<ArticleFeatures> articleFeatures;

	// Here you can load the article features.
	public MyPolicy(String articleFilePath) {
		articleFeatures = new HashSet<ArticleFeatures>();

		Scanner scan = null;
		try {
			scan = new Scanner(new File(articleFilePath));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
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
		
		articleFeatures.add(new ArticleFeatures(articleId, features));
	}

	@Override
	public Article getActionToPerform(User visitor,
			List<Article> possibleActions) {
		return possibleActions.get(0);
	}

	@Override
	public void updatePolicy(User c, Article a, Boolean reward) {
	}
}
