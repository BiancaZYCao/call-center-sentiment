// Example data - this could come from an API or other dynamic source
const topics = [
    {text: 'payment issue', type: 'danger'},
    {text: 'account setup', type: 'primary'},
    {text: 'Product and Service Inquiries', type: 'info'},
    {text: 'Connectivity Issues', type: 'danger'},
    {text: 'Order Tracking and Cancellations', type: 'info'},
    {text: 'Service Restoration', type: 'primary'},
    {text: 'Current Promotions', type: 'success'},
    {text: 'Service Outages and Downtime', type: 'warning'},
    {text: 'Customer Feedback and Complaints', type: 'danger'},
    {text: 'Positive Feedback', type: 'success'},
    {text: 'Suggestions', type: 'secondary'},
    {text: 'Service/Product Complaints', type: 'danger'},
    {text: 'Subscription and Membership Benefits', type: 'dark'}
];

// Function to create a tag element
function createTagElement(topic) {
    const tag = document.createElement('div');
    tag.className = `tag tag-${topic.type}`;
    tag.innerHTML = `${topic.text} <span class="close">&times;</span>`;

    // Add event listener for close button
    tag.querySelector('.close').addEventListener('click', function () {
        tag.style.display = 'none';
    });

    return tag;
}

// Function to render tags
function renderTags(topics) {
    const tagsContainer = document.getElementById('tags-container');
    tagsContainer.innerHTML = ''; // Clear any existing tags

    topics.forEach(topic => {
        const tagElement = createTagElement(topic);
        tagsContainer.appendChild(tagElement);
    });
}

// Initialize the tags on page load
document.addEventListener('DOMContentLoaded', function() {
    renderTags(topics);
});
