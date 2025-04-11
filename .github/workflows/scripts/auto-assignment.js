/**
 Automatically assign issues and PRs to users in the `assigneesList`
 *  on a rotating basis.

  @param {!object}
    GitHub objects can call GitHub APIs using their built-in library functions.
    The context object contains issue and PR details.
*/

module.exports = async ({github, context}) => {
  let issueNumber;
  let assigneesList;
  // Is this an issue? If so, assign the issue number. Otherwise, assign the PR
  // number.
  if (context.payload.issue) {
    assigneesList = ['pkgoogle', 'gaikwadrahul8'];  // for issues
    issueNumber = context.payload.issue.number;
  } else {
    assigneesList = [];  // for PRs
    issueNumber = context.payload.number;
  }
  console.log('assignee list', assigneesList);
  console.log('entered auto assignment for this issue:  ', issueNumber);
  if (!assigneesList.length) {
    console.log('No assignees found for this repo.');
    return;
  }
  let noOfAssignees = assigneesList.length;
  let selection = issueNumber % noOfAssignees;
  let assigneeForIssue = assigneesList[selection];

  console.log(
      'issue Number = ', issueNumber + ' , assigning to: ', assigneeForIssue);
  return github.rest.issues.addAssignees({
    issue_number: context.issue.number,
    owner: context.repo.owner,
    repo: context.repo.repo,
    assignees: [assigneeForIssue],
  });
};
